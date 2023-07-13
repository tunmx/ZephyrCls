from zephyrcls.command.aliased_group import AliasedGroup
from zephyrcls.command.train import train
from zephyrcls.command.evaluate import evaluate
from zephyrcls.command.export import export
from zephyrcls.command.visualized_data import visualized
import click


__all__ = ['cli']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(export)
cli.add_command(visualized)

if __name__ == '__main__':
    cli()