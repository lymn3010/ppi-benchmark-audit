"""Explicit dependencies common to realization channels."""
from .arguments import ArgumentBuilder


class EventChannel:
    parser_name = "dependency"

    def __init__(self, resources=None, context=None):
        self.resources = resources
        self.context = context
        self.arguments = ArgumentBuilder(resources, context)
