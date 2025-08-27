from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def print_summary_to_cli(summary: dict):
    text = ""
    for section, details in summary.items():
        text += f"{section}:\n"
        for key, val in details.items():
            text += f"  - {key}: {val}\n"
    markdown = Markdown(text)
    console.print(Panel.fit(markdown, title="🧾 Summary Report", border_style="green"))
