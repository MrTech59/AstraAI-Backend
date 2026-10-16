"""Astra CLI.

Usage:
    python -m cli.main chat       (interactive session)
    python -m cli.main --version  (print version)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root is on path when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import astra
from astra.config import filesystem_roots, load_settings, memory_db_path
from astra.core import AgentLoop, ContextBuilder
from astra.logging import setup_logging
from astra.memory import MemoryStore
from astra.mcp.client import connect_mcp_servers
from astra.permissions import CliApprovalProvider, PermissionManager, PermissionPolicy
from astra.providers import create_provider
from astra.security import ShellPolicy
from astra.setup import SetupError, default_answers, run_setup
from astra.tools import ToolRegistry, make_filesystem_tools, make_shell_tool


def get_version() -> str:
    return f"Astra Core {astra.__version__}"


def print_help() -> None:
    print("""
Astra Core CLI commands
  /help       Show this help
  /tools      List available tools
  /memory     Show recent memories
  /status     Show active configuration
  /exit       Quit
""")


def print_status(settings) -> None:
    print(f"Provider: {settings.llm.provider}")
    print(f"Model: {settings.llm.model}")
    print(f"Base URL: {settings.llm.base_url}")
    print(f"Shell allowed: {settings.security.allow_shell}")


async def interactive_chat() -> int:
    settings = load_settings()
    logger = setup_logging(settings.logging.level)
    logger.info("%s starting", get_version())

    try:
        provider = create_provider(settings.llm)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tools = ToolRegistry()
    roots = filesystem_roots(settings) or [Path.cwd()]
    for tool in make_filesystem_tools(roots=roots):
        tools.register(tool)

    if settings.security.allow_shell:
        policy = ShellPolicy(
            allow_shell=True,
            allowed_prefixes=settings.security.allowed_shell_prefixes,
        )
        tools.register(make_shell_tool(policy))

    mcp_tools = await connect_mcp_servers(settings.mcp_servers)
    for tool in mcp_tools:
        tools.register(tool)

    memory = MemoryStore(memory_db_path(settings))
    context = ContextBuilder(
        memory=memory,
        system_prompt=settings.agent.system_prompt,
        max_messages=settings.context.max_messages,
        max_memories=settings.context.max_memories,
    )
    permissions = PermissionManager(
        policy=PermissionPolicy.from_config(settings.permissions),
        approval=CliApprovalProvider(),
        registry=tools,
    )
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        context=context,
        max_iterations=settings.agent.max_tool_iterations,
        memory=memory,
        permissions=permissions,
    )

    logger.info("Provider: %s", settings.llm.provider)
    logger.info("Model: %s", settings.llm.model)
    logger.info("Loaded %s tools", len(tools))

    print(get_version())
    print(f"Provider: {settings.llm.provider}")
    print(f"Model: {settings.llm.model}")
    print("Type /help for commands or /exit to quit.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in {"/exit", ":q", "quit", "exit"}:
                break
            if user_input.lower() == "/help":
                print_help()
                continue
            if user_input.lower() == "/tools":
                for name in tools.names():
                    print(f"  - {name}")
                continue
            if user_input.lower() == "/memory":
                recent = memory.list_memories(10)
                if not recent:
                    print("  (no memories)")
                for m in recent:
                    print(f"  [{m.id}] {m.content[:80]}")
                continue
            if user_input.lower() == "/status":
                print_status(settings)
                continue

            print()
            try:
                reply = await agent.run(user_input)
                print(f"Astra: {reply}\n")
            except Exception as exc:
                logger.exception("Turn failed")
                print(f"Error: {exc}\n")
    finally:
        memory.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Astra Core CLI")
    parser.add_argument("--version", action="store_true", help="show version and exit")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", help="interactive chat session (default)")

    setup_parser = sub.add_parser("setup", help="interactive first-time setup")
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be written without creating files",
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="use default answers instead of prompting (useful with --dry-run)",
    )
    setup_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="directory to write config.toml and .env into (default: cwd)",
    )
    args = parser.parse_args()

    if args.version:
        print(get_version())
        return 0

    if args.command == "setup":
        try:
            if args.non_interactive:
                run_setup(
                    root_dir=args.root,
                    answers=default_answers(),
                    dry_run=args.dry_run,
                )
            else:
                run_setup(root_dir=args.root, dry_run=args.dry_run)
        except SetupError as exc:
            print(f"Setup error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command is None or args.command == "chat":
        return asyncio.run(interactive_chat())

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
