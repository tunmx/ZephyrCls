from zephyrcls.command.aliased_group import AliasedGroup
from zephyrcls.command.train import train
import click


__all__ = ['cli']

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(train)

if __name__ == '__main__':
    cli()