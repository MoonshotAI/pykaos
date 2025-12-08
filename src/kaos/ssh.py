from __future__ import annotations

import os
import shlex
from asyncio import StreamReader, StreamWriter
from collections.abc import AsyncGenerator
from pathlib import PurePath, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

import asyncssh

from kaos import Kaos, StatResult, StrOrKaosPath
from kaos.path import KaosPath

if TYPE_CHECKING:

    def type_check(ssh: SSHKaos) -> None:
        _: Kaos = ssh


class SSHKaos:
    """
    A KAOS implementation that interacts with a remote machine via SSH.
    """

    name: str = "ssh"

    class Process(Kaos.Process):
        """KAOS process wrapper around asyncssh.SSHClientProcess."""

        def __init__(self, process: asyncssh.SSHClientProcess[bytes]) -> None:
            self._process = process
            self.stdin: StreamWriter = cast(StreamWriter, process.stdin)
            self.stdout: StreamReader = cast(StreamReader, process.stdout)
            self.stderr: StreamReader = cast(StreamReader, process.stderr)

        @property
        def pid(self) -> int:
            # FIXME: SSHClientProcess does not have a pid attribute.
            return -1

        @property
        def returncode(self) -> int | None:
            return self._process.returncode

        async def wait(self) -> int:
            completed = await self._process.wait()
            if completed.returncode is not None:
                return completed.returncode
            return self._process.returncode or 0

        async def kill(self) -> None:
            self._process.kill()

    def __init__(
        self,
        *,
        host: str,
        port: int = 22,
        username: str | None = None,
        password: str | None = None,
        key_paths: list[str] | None = None,
        key_contents: list[str] | None = None,
        **options: object,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_paths = key_paths
        self.key_contents = key_contents
        self.options = options

        self._connection: asyncssh.SSHClientConnection | None = None
        self._sftp: asyncssh.SFTPClient | None = None
        self._home_dir: str | None = None
        self._cwd: str = "."

    async def _ensure_connected(self) -> None:
        """Ensure SSH connection is established."""
        if self._connection is None:
            options = {**self.options}
            if self.username:
                options["username"] = self.username
            if self.password:
                options["password"] = self.password
            client_keys: list[str | asyncssh.SSHKey] = []
            if self.key_contents:
                client_keys.extend(
                    [asyncssh.import_private_key(key) for key in self.key_contents]
                )
            if self.key_paths:
                client_keys.extend(self.key_paths)
            if client_keys:
                options["client_keys"] = client_keys

            try:
                self._connection = await asyncssh.connect(
                    self.host,
                    port=self.port,
                    encoding=None,
                    **options,
                )
                self._sftp = await self._connection.start_sftp_client()
            except Exception:
                self._connection = None
                self._sftp = None
                raise

    async def _get_home_dir(self) -> str:
        """Get the home directory on the remote server."""
        if self._home_dir is None:
            await self._ensure_connected()
            assert self._connection is not None

            result = await self._connection.run("echo $HOME")
            if result.returncode == 0 and result.stdout:
                if isinstance(result.stdout, bytes):
                    self._home_dir = result.stdout.decode("utf-8").strip()
                else:
                    self._home_dir = result.stdout.strip()
            else:
                # Fallback to /home/<username>
                self._home_dir = f"/home/{self.username}" if self.username else "/"

        return self._home_dir

    async def _expand_home(self, path: str) -> str:
        """Expand a leading ~ to the remote home directory."""
        if path.startswith("~"):
            home = await self._get_home_dir()
            return path.replace("~", home, 1)
        return path

    async def _resolve_path(self, path: str) -> str:
        """Resolve a path, handling ~ and relative paths."""
        await self._ensure_connected()

        expanded = await self._expand_home(str(path))
        if os.path.isabs(expanded):
            return os.path.normpath(expanded)

        base = await self._expand_home(self._cwd)
        return os.path.normpath(os.path.join(base, expanded))

    async def _ssh_run(self, *args: str) -> tuple[int, str, str]:
        """Execute a command on the remote server and return the return code, stdout, and stderr."""
        if not args:
            raise ValueError("At least one argument is required for _ssh_exec.")

        await self._ensure_connected()
        assert self._connection is not None

        command = " ".join(shlex.quote(arg) for arg in args)

        result = await self._connection.run(command, encoding=None)
        if result.stdout:
            if isinstance(result.stdout, bytes):
                stdout = result.stdout.decode("utf-8", "ignore").strip()
            else:
                stdout = result.stdout.strip()
        else:
            stdout = ""
        if result.stderr:
            if isinstance(result.stderr, bytes):
                stderr = result.stderr.decode("utf-8", "ignore").strip()
            else:
                stderr = result.stderr.strip()
        else:
            stderr = ""

        return result.returncode or 0, stdout, stderr

    def pathclass(self) -> type[PurePath]:
        return PurePosixPath

    def gethome(self) -> KaosPath:
        try:
            home = self._home_dir or "~"
            return KaosPath(home)
        except Exception:
            return KaosPath("~")

    def getcwd(self) -> KaosPath:
        return KaosPath(self._cwd)

    async def chdir(self, path: StrOrKaosPath) -> None:
        path_str = str(path)
        resolved_path = await self._resolve_path(path_str)
        # Verify it's a directory
        try:
            assert self._sftp
            stat_result = await self._sftp.stat(resolved_path)
            if not stat_result:
                raise NotADirectoryError(f"Not a directory: {path_str}")
            if stat_result.type != asyncssh.FILEXFER_TYPE_DIRECTORY:
                raise NotADirectoryError(f"Not a directory: {path_str}")
        except asyncssh.SFTPError as e:
            raise OSError(f"chdir failed: {e}") from e

        self._cwd = resolved_path

    async def stat(
        self,
        path: StrOrKaosPath,
        *,
        follow_symlinks: bool = True,
    ) -> StatResult:
        resolved_path = await self._resolve_path(str(path))

        fmt = "%f %u %g %s %X %Y %Z %h %i %D"
        args = ["stat"]
        if follow_symlinks:
            args.append("-L")
        args += ["-c", fmt, resolved_path]

        returncode, stdout, stderr = await self._ssh_run(*args)
        if returncode != 0:
            raise OSError(f"stat failed: {stderr}")

        tokens = stdout.split()
        if len(tokens) != 10:
            raise OSError(f"stat returned unexpected output: {stdout!r}")

        (
            st_mode_hex,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_nlink,
            st_ino,
            st_dev,
        ) = tokens

        return StatResult(
            st_mode=int(st_mode_hex, 16),
            st_uid=int(st_uid),
            st_gid=int(st_gid),
            st_size=int(st_size),
            st_atime=float(st_atime),
            st_mtime=float(st_mtime),
            st_ctime=float(st_ctime),
            st_ino=int(st_ino),
            st_dev=int(st_dev, 16),
            st_nlink=int(st_nlink),
        )

    async def iterdir(self, path: StrOrKaosPath) -> AsyncGenerator[KaosPath]:
        await self._ensure_connected()
        assert self._sftp is not None

        resolved_path = await self._resolve_path(str(path))

        try:
            entries = await self._sftp.listdir(resolved_path)
            for entry in entries:
                # NOTE: sftp listdir gives . and ..
                if entry in {".", ".."}:
                    continue
                yield KaosPath(entry)
        except asyncssh.SFTPError as e:
            raise OSError(f"iterdir failed: {e}") from e

    async def glob(
        self,
        path: StrOrKaosPath,
        pattern: str,
        *,
        case_sensitive: bool = True,
    ) -> AsyncGenerator[KaosPath]:
        resolved_path = await self._resolve_path(str(path))

        name_flag = "-name" if case_sensitive else "-iname"
        cmd = [
            "find",
            resolved_path,
            name_flag,
            pattern,
        ]

        returncode, stdout, stderr = await self._ssh_run(*cmd)
        if returncode != 0:
            raise OSError(f"glob failed: {stderr}")
        for line in stdout.splitlines():
            if line:
                yield KaosPath(line)

    async def readbytes(self, path: StrOrKaosPath) -> bytes:
        await self._ensure_connected()
        assert self._sftp is not None

        resolved_path = await self._resolve_path(str(path))

        try:
            async with self._sftp.open(resolved_path, "rb") as f:
                return await f.read()
        except asyncssh.SFTPError as e:
            raise OSError(f"readbytes failed: {e}") from e

    async def readtext(
        self,
        path: str | KaosPath,
        *,
        encoding: str = "utf-8",
        errors: Literal["strict", "ignore", "replace"] = "strict",
    ) -> str:
        data = await self.readbytes(path)
        return data.decode(encoding, errors=errors)

    async def readlines(
        self,
        path: str | KaosPath,
        *,
        encoding: str = "utf-8",
        errors: Literal["strict", "ignore", "replace"] = "strict",
    ) -> AsyncGenerator[str]:
        # NOTE: readlines is not supported by SFTPClientFile
        text = await self.readtext(path, encoding=encoding, errors=errors)
        for line in text.splitlines():
            yield line

    async def writebytes(self, path: StrOrKaosPath, data: bytes) -> int:
        await self._ensure_connected()
        assert self._sftp is not None

        resolved_path = await self._resolve_path(str(path))

        try:
            async with self._sftp.open(resolved_path, "wb") as f:
                return await f.write(data)
        except asyncssh.SFTPError as e:
            raise OSError(f"writebytes failed: {e}") from e

    async def writetext(
        self,
        path: str | KaosPath,
        data: str,
        *,
        mode: Literal["w"] | Literal["a"] = "w",
        encoding: str = "utf-8",
        errors: Literal["strict", "ignore", "replace"] = "strict",
    ) -> int:
        encoded_data = data.encode(encoding, errors=errors)
        await self._ensure_connected()
        assert self._sftp is not None

        resolved_path = await self._resolve_path(str(path))

        try:
            async with self._sftp.open(resolved_path, f"{mode}b") as f:
                return await f.write(encoded_data)
        except asyncssh.SFTPError as e:
            raise OSError(f"writetext failed: {e}") from e

    async def mkdir(
        self,
        path: StrOrKaosPath,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        await self._ensure_connected()
        resolved_path = await self._resolve_path(str(path))

        if not exist_ok:
            try:
                assert self._sftp is not None
                exists = await self._sftp.exists(resolved_path)
                if exists:
                    raise FileExistsError(f"{resolved_path} already exists")
            except asyncssh.SFTPError as e:
                raise OSError(f"mkdir failed: {e}") from e
            except Exception as e:
                raise OSError(f"mkdir failed: {e}") from e

        cmd = ["mkdir"]
        if parents or exist_ok:
            cmd.append("-p")
        cmd.append(resolved_path)

        returncode, _, stderr = await self._ssh_run(*cmd)
        if returncode != 0:
            raise OSError(f"mkdir failed: {stderr}")

    async def exec(self, *args: str) -> Kaos.Process:
        if not args:
            raise ValueError(
                "At least one argument (the program to execute) is required."
            )

        await self._ensure_connected()
        assert self._connection is not None

        command = " ".join(shlex.quote(arg) for arg in args)
        process = await self._connection.create_process(command, encoding=None)
        return self.Process(process)

    async def close(self) -> None:
        """Close the SSH connection."""
        if self._sftp:
            self._sftp.exit()
            self._sftp = None
        if self._connection:
            self._connection.close()
            self._connection = None
