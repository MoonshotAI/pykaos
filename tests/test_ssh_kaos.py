from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import AsyncGenerator
from pathlib import PurePosixPath
from uuid import uuid4

import pytest

from kaos import reset_current_kaos, set_current_kaos
from kaos.path import KaosPath
from kaos.ssh import SSHKaos


@pytest.fixture
def ssh_kaos_config() -> dict[str, str | None]:
    """SSH 连接参数，从环境变量读取。"""
    return {
        "host": os.environ.get("KAOS_SSH_HOST", "localhost"),
        "port": os.environ.get("KAOS_SSH_PORT", "22"),
        "username": os.environ.get("KAOS_SSH_USERNAME"),
        "password": os.environ.get("KAOS_SSH_PASSWORD"),
        "key_paths": os.environ.get("KAOS_SSH_KEY_PATHS"),
    }


@pytest.fixture
async def ssh_kaos(ssh_kaos_config: dict[str, str | None]) -> AsyncGenerator[SSHKaos | None]:
    """若配置齐全则创建 SSH KAOS 实例，否则跳过。"""
    config = ssh_kaos_config
    if not config["host"] or not config["username"]:
        pytest.skip("SSH test configuration not provided")

    key_paths = config.get("key_paths")
    kaos = SSHKaos(
        host=str(config["host"]),
        username=config["username"],
        port=int(config["port"] or 22),
        password=config.get("password"),
        key_paths=key_paths.split(",") if key_paths else None,
    )

    try:
        yield kaos
    except Exception:
        pytest.skip("SSH connection failed")
    finally:
        await kaos.close()


@pytest.fixture
async def remote_base(ssh_kaos: SSHKaos) -> AsyncGenerator[str, None]:
    """为每个用例准备并清理远程工作目录。"""
    base = f"/tmp/kaos_test_{os.getpid()}_{uuid4().hex}"
    await ssh_kaos.mkdir(base, parents=True, exist_ok=True)

    try:
        yield base
    finally:
        try:
            proc = await ssh_kaos.exec("rm", "-rf", base)
            await proc.wait()
        except Exception:
            pass
        try:
            await ssh_kaos.chdir("~")
        except Exception:
            pass


@pytest.fixture
def with_current_kaos(ssh_kaos: SSHKaos):
    """将当前 KAOS 设置为 SSH 实例，便于 KaosPath 相关 API 联动。"""
    token = set_current_kaos(ssh_kaos)
    try:
        yield ssh_kaos
    finally:
        reset_current_kaos(token)


@pytest.mark.asyncio
async def test_pathclass_and_initial_cwd(ssh_kaos: SSHKaos | None):
    if ssh_kaos is None:
        pytest.skip("SSH not available")

    assert ssh_kaos.pathclass() is PurePosixPath
    assert isinstance(ssh_kaos.gethome(), KaosPath)
    assert str(ssh_kaos.getcwd()) == "."


@pytest.mark.asyncio
async def test_chdir_and_relative_resolution(ssh_kaos: SSHKaos, remote_base: str):
    await ssh_kaos.chdir(remote_base)
    assert str(ssh_kaos.getcwd()) == remote_base

    await ssh_kaos.mkdir("child", exist_ok=True)
    await ssh_kaos.chdir("child")
    assert str(ssh_kaos.getcwd()) == os.path.join(remote_base, "child")

    await ssh_kaos.chdir("..")
    assert str(ssh_kaos.getcwd()) == remote_base


@pytest.mark.asyncio
async def test_mkdir_variants_and_stat(ssh_kaos: SSHKaos, remote_base: str):
    nested_dir = os.path.join(remote_base, "deep/level")
    await ssh_kaos.mkdir(nested_dir, parents=True, exist_ok=False)

    stat_result = await ssh_kaos.stat(nested_dir, follow_symlinks=False)
    assert stat.S_ISDIR(stat_result.st_mode)

    await ssh_kaos.mkdir(nested_dir, parents=True, exist_ok=True)

    file_path = os.path.join(nested_dir, "file.txt")
    content = "mkdir + stat"
    await ssh_kaos.writetext(file_path, content)
    file_stat = await ssh_kaos.stat(file_path)
    assert file_stat.st_size == len(content)


@pytest.mark.asyncio
async def test_write_and_read_text_with_kaospath(with_current_kaos: SSHKaos, remote_base: str):
    await with_current_kaos.chdir(remote_base)

    path = KaosPath(remote_base) / "text.txt"
    data = "Hello SSH\n"
    appended = "More\n"

    written = await path.write_text(data)
    assert written == len(data)

    appended_len = await path.append_text(appended)
    assert appended_len == len(appended)

    full = await path.read_text()
    assert full == data + appended

    lines = [line async for line in path.read_lines()]
    assert lines == ["Hello SSH", "More"]

    assert str(KaosPath.cwd()) == remote_base


@pytest.mark.asyncio
async def test_write_and_read_bytes(ssh_kaos: SSHKaos, remote_base: str):
    binary_path = os.path.join(remote_base, "bin.dat")
    payload = bytes(range(16))

    size = await ssh_kaos.writebytes(binary_path, payload)
    assert size == len(payload)

    data = await ssh_kaos.readbytes(binary_path)
    assert data == payload

    stat_result = await ssh_kaos.stat(binary_path)
    assert stat_result.st_size == len(payload)


@pytest.mark.asyncio
async def test_iterdir_and_glob(ssh_kaos: SSHKaos, remote_base: str):
    await ssh_kaos.writetext(os.path.join(remote_base, "file1.txt"), "1")
    await ssh_kaos.writetext(os.path.join(remote_base, "file2.log"), "2")
    await ssh_kaos.mkdir(os.path.join(remote_base, "sub"), exist_ok=True)
    await ssh_kaos.writetext(os.path.join(remote_base, "sub", "inner.LOG"), "3")

    entries = {entry.name async for entry in ssh_kaos.iterdir(remote_base)}
    assert entries == {"file1.txt", "file2.log", "sub"}

    logs = {str(path) async for path in ssh_kaos.glob(remote_base, "*.log", case_sensitive=False)}
    assert {
        os.path.join(remote_base, "file2.log"),
        os.path.join(remote_base, "sub", "inner.LOG"),
    } <= logs


@pytest.mark.asyncio
async def test_glob_case_sensitivity(ssh_kaos: SSHKaos, remote_base: str):
    lower = os.path.join(remote_base, "lower.log")
    upper = os.path.join(remote_base, "Upper.LOG")
    mixed = os.path.join(remote_base, "MiXeD.LoG")

    await ssh_kaos.writetext(lower, "a")
    await ssh_kaos.writetext(upper, "b")
    await ssh_kaos.writetext(mixed, "c")

    insensitive = {
        str(path) async for path in ssh_kaos.glob(remote_base, "*.log", case_sensitive=False)
    }
    assert {lower, upper, mixed} <= insensitive

    sensitive = {
        str(path) async for path in ssh_kaos.glob(remote_base, "*.log", case_sensitive=True)
    }
    assert lower in sensitive
    assert upper not in sensitive
    assert mixed not in sensitive


@pytest.mark.asyncio
async def test_glob_directories(ssh_kaos: SSHKaos, remote_base: str):
    lower_dir = os.path.join(remote_base, "data")
    upper_dir = os.path.join(remote_base, "DataSet")
    await ssh_kaos.mkdir(lower_dir, exist_ok=True)
    await ssh_kaos.mkdir(upper_dir, exist_ok=True)
    await ssh_kaos.writetext(os.path.join(lower_dir, "file.txt"), "1")
    await ssh_kaos.writetext(os.path.join(upper_dir, "inner.log"), "2")

    insensitive_dirs = {
        str(path) async for path in ssh_kaos.glob(remote_base, "data*", case_sensitive=False)
    }
    assert lower_dir in insensitive_dirs
    assert upper_dir in insensitive_dirs

    sensitive_dirs = {
        str(path) async for path in ssh_kaos.glob(remote_base, "data*", case_sensitive=True)
    }
    assert lower_dir in sensitive_dirs
    assert upper_dir not in sensitive_dirs


@pytest.mark.asyncio
async def test_glob_directory_case_patterns(ssh_kaos: SSHKaos, remote_base: str):
    lower_dir = os.path.join(remote_base, "logs")
    upper_dir = os.path.join(remote_base, "LOGS")
    mixed_dir = os.path.join(remote_base, "LogFiles")
    nested_dir = os.path.join(upper_dir, "Archive")

    for path in (lower_dir, upper_dir, mixed_dir, nested_dir):
        await ssh_kaos.mkdir(path, parents=True, exist_ok=True)

    insensitive = {
        str(path) async for path in ssh_kaos.glob(remote_base, "log*", case_sensitive=False)
    }
    assert {lower_dir, upper_dir, mixed_dir} <= insensitive

    sensitive_lower = {
        str(path) async for path in ssh_kaos.glob(remote_base, "log*", case_sensitive=True)
    }
    assert lower_dir in sensitive_lower
    assert upper_dir not in sensitive_lower
    assert mixed_dir not in sensitive_lower

    nested_insensitive = {
        str(path) async for path in ssh_kaos.glob(upper_dir, "arch*", case_sensitive=False)
    }
    assert nested_dir in nested_insensitive

    nested_sensitive = {
        str(path) async for path in ssh_kaos.glob(upper_dir, "arch*", case_sensitive=True)
    }
    assert nested_dir not in nested_sensitive


@pytest.mark.asyncio
async def test_exec_stdout_and_stderr(ssh_kaos: SSHKaos | None):
    if ssh_kaos is None:
        pytest.skip("SSH not available")

    proc = await ssh_kaos.exec("sh", "-c", "echo out && echo err 1>&2")
    stdout_data, stderr_data = await asyncio.gather(proc.stdout.read(), proc.stderr.read())
    code = await proc.wait()

    assert code == 0
    assert stdout_data.decode().strip() == "out"
    assert stderr_data.decode().strip() == "err"


@pytest.mark.asyncio
async def test_exec_kill_and_pid(ssh_kaos: SSHKaos | None):
    if ssh_kaos is None:
        pytest.skip("SSH not available")

    proc = await ssh_kaos.exec("sh", "-c", "echo ready; sleep 5")
    first_line = await proc.stdout.readline()
    assert first_line.decode().strip() == "ready"

    await proc.kill()
    exit_code = await proc.wait()
    assert exit_code != 0
    assert isinstance(proc.pid, int)


@pytest.mark.asyncio
async def test_close_and_reconnect(ssh_kaos: SSHKaos, remote_base: str):
    first_file = os.path.join(remote_base, "before_close.txt")
    await ssh_kaos.writetext(first_file, "keep")

    await ssh_kaos.close()

    second_file = os.path.join(remote_base, "after_close.txt")
    written = await ssh_kaos.writetext(second_file, "reconnected")
    assert written == len("reconnected")

    content = await ssh_kaos.readtext(second_file)
    assert content == "reconnected"
