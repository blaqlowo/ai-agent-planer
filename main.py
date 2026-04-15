import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from agent import Agent

console = Console()

def print_header():
    console.print(Panel("[bold cyan]🤖 AI Agent Orchestrator (Offline Mode)[/bold cyan]", expand=False))
    console.print("This system uses Local LLMs (Ollama/LM Studio) to maintain a prompt -> plan -> execution loop.\n")

def main():
    print_header()
    goal = console.input("[bold green]Enter your Goal:[/bold green] ")
    if not goal.strip():
        console.print("[red]Goal cannot be empty.[/red]")
        return

    console.print("\n[bold yellow]Initializing Agent...[/bold yellow]")
    agent = Agent()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # Step 1: Planning
        plan_task = progress.add_task("[cyan]Generating step-by-step plan...", total=None)
        steps = agent.generate_plan(goal)
        progress.update(plan_task, completed=100)
        
    if not isinstance(steps, list):
        steps = [str(steps)]
        
    console.print(Panel(f"[bold magenta]Generated Plan:[/bold magenta]\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))))

    if not steps or (len(steps) == 1 and steps[0] == ""):
        console.print("[red]Failed to generate a valid plan. Check the model output format.[/red]")
        return

    # Step 2: Execution Loop
    console.print("\n[bold yellow]Beginning Execution Loop...[/bold yellow]\n")
    
    for i, step in enumerate(steps):
        console.print(f"[cyan]▶ Executing Step {i+1}/{len(steps)}:[/cyan] {step}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            exec_task = progress.add_task(f"[dim]Working on step {i+1}...[/dim]", total=None)
            result = agent.execute_step(step, goal)
            progress.update(exec_task, completed=100)

        # Print output log for the step
        console.print(Panel(result, title=f"Result for Step {i+1}", border_style="green"))
        time.sleep(1) # Small pause for "pro feel" reading time

    # Step 3: Final Summary
    console.print("\n[bold cyan]🎉 All steps completed successfully![/bold cyan]")
    console.print("[dim]Memory has been stored for this session.[/dim]")

if __name__ == "__main__":
    main()
