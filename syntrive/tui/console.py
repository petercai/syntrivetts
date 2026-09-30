import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markup import escape as markup_escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.syntax import Syntax
from rich import box


_BANNER = """
 ╔═══════════════════════════════╗
 ║   SyntriveTTS  v0.4.0 (dev)  ║
 ║   EPUB → Audiobook Pipeline  ║
 ╚═══════════════════════════════╝
"""


class SyntriveConsole:
    def __init__(self, console: Optional[Console] = None) -> None:
        self._c = console or Console()

    def show_banner(self) -> None:
        self._c.print(_BANNER, style="bold cyan")

    def show_step_summary(
        self,
        step_name: str,
        step_number: int,
        total_steps: int,
        tasks: list[str],
        estimated_seconds: Optional[int] = None,
    ) -> None:
        lines = [f"[bold]Tasks:[/bold]"]
        for task in tasks:
            lines.append(f"  • {task}")
        if estimated_seconds:
            lines.append(f"\n[dim]Estimated time: ~{estimated_seconds}s[/dim]")

        body = "\n".join(lines)
        self._c.print(
            Panel(
                body,
                title=f"[bold cyan]STEP {step_number}/{total_steps}: {step_name}[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )

    def show_step_result(
        self,
        step_name: str,
        artifacts: dict[str, str],
        notes: Optional[str] = None,
    ) -> None:
        lines = ["[bold]Outputs:[/bold]"]
        for key, value in artifacts.items():
            lines.append(f"  [green]✓[/green] {key}: {value}")
        if notes:
            lines.append(f"\n[italic]{markup_escape(notes)}[/italic]")

        self._c.print(
            Panel(
                "\n".join(lines),
                title=f"[bold green]✓ {step_name} complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    def show_step_error(self, step_name: str, error: str) -> None:
        self._c.print(
            Panel(
                f"[red]{error}[/red]",
                title=f"[bold red]✗ {step_name} failed[/bold red]",
                border_style="red",
                expand=False,
            )
        )

    def show_job_table(self, jobs: list) -> None:
        table = Table(
            title="SyntriveTTS — All Jobs",
            box=box.ROUNDED,
            show_lines=True,
            highlight=True,
        )
        table.add_column("ID", style="bold", width=5)
        table.add_column("Book", style="cyan")
        table.add_column("Stage", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Created", style="dim")

        _status_colors = {
            "pending": "yellow",
            "running": "cyan",
            "completed": "green",
            "done": "green",
            "failed": "red",
            "blocked": "red",
            "paused": "yellow",
        }

        for job in jobs:
            color = _status_colors.get(job.status, "white")
            table.add_row(
                str(job.id),
                getattr(job, "book", None) and job.book.title or "—",
                job.stage or "—",
                f"[{color}]{job.status}[/{color}]",
                str(job.created_at)[:10] if job.created_at else "—",
            )

        self._c.print(table)

    def show_incomplete_jobs_table(self, jobs: list) -> None:
        if not jobs:
            self._c.print("[yellow]No incomplete jobs found.[/yellow]")
            return
        self.show_job_table(jobs)

    def ask_confirm(self, prompt: str = "Proceed?") -> bool:
        return Confirm.ask(prompt)

    def ask_choice(self, prompt: str, choices: list[str]) -> str:
        for i, ch in enumerate(choices, 1):
            self._c.print(f"  [{i}] {ch}")
        self._c.print(f"  [Q] Save & quit")
        raw = Prompt.ask(prompt, choices=[str(i) for i in range(1, len(choices) + 1)] + ["q", "Q"])
        if raw.lower() == "q":
            return "quit"
        return choices[int(raw) - 1]

    def ask_select_job(self, jobs: list):
        self.show_incomplete_jobs_table(jobs)
        if not jobs:
            return None
        ids = [str(j.id) for j in jobs]
        raw = Prompt.ask("Enter job ID to resume (q to quit)", choices=ids + ["q", "Q"])
        if raw.lower() == "q":
            return None
        return next((j for j in jobs if str(j.id) == raw), None)

    def ask_next_step(self, options: list[str]) -> str:
        return self.ask_choice("Select next step", options)

    def make_progress(self, description: str = "Processing...") -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{description}[/cyan]"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self._c,
        )

    def show_file(self, path: Path) -> None:
        if not path.is_file():
            self._c.print(f"[red]File not found:[/red] {path}")
            return

        suffix = path.suffix.lower()
        lexer = (
            "html" if suffix in (".html", ".htm")
            else "json" if suffix == ".json"
            else "text"
        )
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            syntax = Syntax(content, lexer, line_numbers=True, word_wrap=True)
            self._c.print(Panel(syntax, title=str(path), border_style="dim"))
        except Exception as exc:
            self._c.print(f"[red]Cannot read file: {exc}[/red]")

    def info(self, message: str) -> None:
        self._c.print(f"[cyan]ℹ[/cyan] {message}")

    def success(self, message: str) -> None:
        self._c.print(f"[green]✓[/green] {message}")

    def warning(self, message: str) -> None:
        self._c.print(f"[yellow]⚠[/yellow] {message}")

    def error(self, message: str) -> None:
        self._c.print(f"[red]✗[/red] {message}", file=sys.stderr)
