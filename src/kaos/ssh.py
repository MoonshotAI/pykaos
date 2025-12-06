from __future__ import annotations

import os
import shlex
import stat
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
        host: str,
        username: str,
        port: int = 22,
        password: str | None = None,
        key_filename: str | None = None,
        known_hosts: str | None = None,
    ) -> None:
        self.host = host
        self.username = username
        self.port = port
        self.password = password
        self.key_filename = key_filename
        self.known_hosts = known_hosts

        self._connection: asyncssh.SSHClientConnection | None = None
        self._sftp: asyncssh.SFTPClient | None = None
        self._home_dir: str | None = None
        self._cwd: str = "~"

    async def _ensure_connected(self) -> None:
        """Ensure SSH connection is established."""
        if self._connection is None:
            # Prepare connection options
            options = {}
            if self.username:
                options["username"] = self.username
            if self.password:
                options["password"] = self.password
            if self.key_filename:
                options["client_keys"] = [self.key_filename]
            if self.known_hosts:
                options["known_hosts"] = asyncssh.read_known_hosts(self.known_hosts)

            try:
                self._connection = await asyncssh.connect(
                    self.host,
                    port=self.port,
                    encoding=None,
                    **options,
                )
                self._sftp = await self._connection.start_sftp_client()
            except Exception:
                # Reset connection state on failure
                self._connection = None
                self._sftp = None
                raise

    async def _get_home_dir(self) -> str:
        """Get the home directory on the remote server."""
        if self._home_dir is None:
            await self._ensure_connected()
            assert self._connection is not None

            # Get home directory by running 'echo $HOME'
            result = await self._connection.run("echo $HOME")
            if result.returncode == 0 and result.stdout:
                if isinstance(result.stdout, bytes):
                    self._home_dir = result.stdout.decode("utf-8").strip()
                else:
                    self._home_dir = result.stdout.strip()
            else:
                # Fallback to /home/username
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

    def pathclass(self) -> type[PurePath]:
        return PurePosixPath

    def gethome(self) -> KaosPath:
        # Get the actual home directory path
        try:
            home = self._home_dir or "~"
            return KaosPath(home)
        except:
            return KaosPath("~")

    def getcwd(self) -> KaosPath:
        return KaosPath(self._cwd)

    async def chdir(self, path: StrOrKaosPath) -> None:
        path_str = str(path)
        # Resolve the path
        resolved_path = await self._resolve_path(path_str)
        # Verify it's a directory
        try:
            assert self._sftp
            stat_result = await self._sftp.stat(resolved_path)
            if not stat_result:
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
        await self._ensure_connected()
        assert self._sftp is not None

        resolved_path = await self._resolve_path(str(path))

        try:
            st = await self._sftp.stat(resolved_path, follow_symlinks=follow_symlinks)
        except asyncssh.SFTPError as e:
            raise OSError(f"stat failed: {e}") from e

        mode = int(st.permissions or 0)

        # Some servers set file type bits separately from permissions (SFTPv4+).
        file_type = cast(int | None, getattr(st, "type", None))
        if file_type is not None and stat.S_IFMT(mode) == 0:
            type_bits = {
                asyncssh.FILEXFER_TYPE_DIRECTORY: stat.S_IFDIR,
                asyncssh.FILEXFER_TYPE_SYMLINK: stat.S_IFLNK,
                asyncssh.FILEXFER_TYPE_REGULAR: stat.S_IFREG,
                asyncssh.FILEXFER_TYPE_CHAR_DEVICE: stat.S_IFCHR,
                asyncssh.FILEXFER_TYPE_BLOCK_DEVICE: stat.S_IFBLK,
                asyncssh.FILEXFER_TYPE_FIFO: stat.S_IFIFO,
                asyncssh.FILEXFER_TYPE_SOCKET: stat.S_IFSOCK,
            }.get(file_type)
            if type_bits is not None:
                mode |= type_bits

        ctime = getattr(st, "crtime", None)
        if ctime is None:
            ctime = getattr(st, "ctime", None)
        if ctime is None:
            ctime = st.mtime

        return StatResult(
            st_mode=mode,
            st_uid=int(st.uid or 0),
            st_gid=int(st.gid or 0),
            st_size=int(st.size or 0),
            st_atime=float(st.atime or 0),
            st_mtime=float(st.mtime or 0),
            st_ctime=float(ctime or 0),
            # Fields not supported by SFTP:
            # NOTE: These fields are not supported by SFTP, so we set them to 0.
            st_ino=0,
            st_dev=0,
            st_nlink=int(getattr(st, "nlink", 0) or 0),
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
        await self._ensure_connected()
        assert self._connection is not None

        resolved_path = await self._resolve_path(str(path))

        # Use glob command on remote server
        cmd = f'find "{resolved_path}" -{"" if case_sensitive else "i"}name "{pattern}" -type f'
        result = await self._connection.run(cmd)

        if result.returncode == 0 and result.stdout:
            if isinstance(result.stdout, bytes):
                lines = result.stdout.decode("utf-8").strip().split("\n")
            else:
                lines = result.stdout.strip().split("\n")
            for line in lines:
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
        assert self._sftp is not None

        resolved_path = await self._resolve_path(str(path))

        if parents:
            try:
                if not exist_ok:
                    st = await self._sftp.stat(resolved_path)
                    if stat.S_ISDIR(int(st.permissions or 0)):
                        raise FileExistsError(f"{resolved_path} already exists")
            except asyncssh.SFTPError:
                pass

            assert self._connection is not None
            cmd = f"mkdir -p {shlex.quote(resolved_path)}"
            result = await self._connection.run(cmd)
            if result.returncode != 0:
                msg = result.stderr or result.stdout or ""
                if isinstance(msg, bytes):
                    msg = msg.decode("utf-8", "ignore")
                raise OSError(f"mkdir failed: {msg}".strip())
            return

        try:
            await self._sftp.mkdir(resolved_path)
        except asyncssh.SFTPError as e:
            if exist_ok:
                try:
                    st = await self._sftp.stat(resolved_path)
                    if not stat.S_ISDIR(int(st.permissions or 0)):
                        raise FileExistsError(
                            f"{resolved_path} exists and is not a directory"
                        )
                    return
                except asyncssh.SFTPError:
                    pass
            raise OSError(f"mkdir failed: {e}") from e

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


# Default SSH KAOS instance factory
def create_ssh_kaos(
    host: str,
    username: str,
    port: int = 22,
    password: str | None = None,
    key_filename: str | None = None,
    known_hosts: str | None = None,
) -> SSHKaos:
    """Create an SSH KAOS instance."""
    return SSHKaos(
        host=host,
        username=username,
        port=port,
        password=password,
        key_filename=key_filename,
        known_hosts=known_hosts,
    )
