import click


@click.command(name="init", deprecated=True, hidden=True, short_help="True", help='Simple program that greets NAME for a total of COUNT times.')
@click.option('--name', default='World', help='The name to greet.')
@click.option('--count', default=1, type=int, help='Number of greetings.')
def hello(name, count):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo(f"Hello {name}!")


if __name__ == '__main__':
    hello()

