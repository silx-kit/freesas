"""
Generalized arg parser for freeSAS apps to ensure unified command line API.
"""

__author__ = "Martha Brennich"
__license__ = "MIT"
__copyright__ = "2020, ESRF"
__date__ = "02/09/2026"

import argparse
import sys
from pathlib import Path

from freesas import dated_version as freesas_version

#: Fields every output template understands, with their description
OUTPUT_TEMPLATE_FIELDS = {
    "dirname": "directory of the input file",
    "basename": "name of the input file, without its extension",
}


def parse_unit(unit_input: str) -> str:
    """
    Parser for sloppy acceptance of unit flags.
    Current rules:
    "A" ➔ "Å"
    :param unit_input: unit flag as provided by the user
    :return: cast of user input to known flag if sloppy rule defined,
             else user input.
    """
    if unit_input == "A":  # pylint: disable=R1705
        return "Å"
    else:
        return unit_input


def check_output_template(template: str, **extra_fields) -> None:
    """
    Exit with an explicit message if an output template uses an unknown field
    or a format specification which does not apply to that field.

    Meant to be called once, before any file gets processed, so that a typo in
    the template does not show up after a lengthy computation.

    :param template: the template provided by the user
    :param extra_fields: fields the calling app supports on top of the common
                         ones, as name: sample value. The value matters: it is
                         what the format specification gets checked against,
                         i.e. index=0 for a field used as {index:02d}.
    """
    known = {**dict.fromkeys(OUTPUT_TEMPLATE_FIELDS, ""), **extra_fields}
    try:
        template.format(**known)
    except (KeyError, IndexError) as err:
        sys.exit(f"Invalid output template {template!r}: unknown field {err}")
    except ValueError as err:
        sys.exit(f"Invalid output template {template!r}: {err}")


def format_output_filename(
    template: str, source, mkdir: bool = False, **extra_fields
) -> Path:
    """
    Build the name of an output file by substituting the fields of a template.

    :param template: template of the file name, i.e. "{dirname}/{basename}.out"
    :param source: name or Path of the input file the output is derived from
    :param mkdir: create the directory of the output file if it is missing,
                  which a template pointing outside of the input directory
                  usually needs
    :param extra_fields: additional fields to substitute, i.e. index=3
    :return: Path of the file to write to
    """
    source = Path(source)
    destination = Path(
        template.format(
            dirname=source.parent, basename=source.stem, **extra_fields
        )
    )
    if mkdir:
        destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


class SASParser:
    """
    Wrapper class for argparse ArgumentParser that provides predefined argument.
    """

    usage = ""

    def __init__(self, prog: str, description: str, epilog: str, **kwargs):
        """
        Create parser argparse ArgumentParser
        - standardized usage text
        - standardized verion text
        - verbose and version args added by default

        :param prog:         name of the executable
        :param description:  description param of argparse ArgumentParser
        :param epilog:       epilog param of argparse ArgumentParser
        :param kwargs:       additional kwargs for argparse ArgumentParser
        """

        self.usage = f"{prog} [OPTIONS] FILES "
        version = f"{prog} version {freesas_version.version} from {freesas_version.date}"

        self.parser = argparse.ArgumentParser(
            usage=self.usage, description=description, epilog=epilog, **kwargs
        )
        self.add_argument(
            "-v",
            "--verbose",
            default=0,
            help="switch to verbose mode",
            action="count",
        )
        self.add_argument("-V", "--version", action="version", version=version)

    def parse_args(self, *args, **kwargs):
        """Wrapper for argparse parse_args()"""
        return self.parser.parse_args(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        """Wrapper for argparse add_argument()"""
        self.parser.add_argument(*args, **kwargs)

    def add_file_argument(self, help_text: str):
        """
        Add positional file argument.

        :param help_text: specific help text to be displayed
        """
        self.add_argument("file", metavar="FILE", nargs="+", help=help_text)

    def add_q_unit_argument(self):
        """
        Add default argument for selecting length unit of input data
        between Å and nm. nm is default.
        """
        self.add_argument(
            "-u",
            "--unit",
            action="store",
            choices=["nm", "Å", "A"],
            help="Unit for q: inverse nm or Ångstrom?",
            default="nm",
            type=parse_unit,
        )

    def add_output_filename_argument(self):
        """Add default argument for specifying output format."""
        self.add_argument(
            "-o",
            "--output",
            action="store",
            help="Output filename",
            default=None,
            type=Path,
        )

    def add_output_template_argument(
        self,
        default: str | None,
        extra_fields: dict[str, str] | None = None,
        described_default: str | None = None,
    ):
        """
        Add the -o/--output argument, taking a template for the file name
        instead of a plain file name, so that one output file can be produced
        per input file.

        :param default: template used when the argument is not provided. Pass
                        None when the default depends on the other options, the
                        app is then in charge of picking one.
        :param extra_fields: fields supported on top of the common ones,
                             as name: description
        :param described_default: default advertised in the help text, when it
                                  cannot be given as `default`
        """
        fields = {**OUTPUT_TEMPLATE_FIELDS, **(extra_fields or {})}
        described = ", ".join(
            f"{{{name}}} ({description})" for name, description in fields.items()
        )
        self.add_argument(
            "-o",
            "--output",
            action="store",
            default=default,
            type=str,
            help="Template for the name of the output file. The fields "
            f"{described} are substituted for every processed file. "
            f"Default: {described_default or default}",
        )

    def add_output_data_format(self, *formats: str, default: str | None = None):
        """Add default argument for specifying output format."""
        help_string = "Output format: " + ", ".join(formats)
        self.add_argument(
            "-f",
            "--format",
            action="store",
            help=help_string,
            default=default,
            type=str,
        )


class GuinierParser:
    """
    Wrapper class for argparse ArgumentParser that provides predefined
    arguments for auto_rg like programs.
    """

    usage = ""

    def __init__(self, prog: str, description: str, epilog: str, **kwargs):
        """
        Create parser argparse ArgumentParser with argument
        - standardized usage text
        - standardized version text
        - verbose and version args added by default

        :param prog:         name of the executable
        :param description:  description param of argparse ArgumentParser
        :param epilog:       epilog param of argparse ArgumentParser
        :param kwargs:       additional kwargs for argparse ArgumentParser
        """

        file_help_text = "dat files of the scattering curves"
        self.parser = SASParser(
            prog=prog, description=description, epilog=epilog, **kwargs
        )
        self.parser.add_file_argument(help_text=file_help_text)
        self.parser.add_output_filename_argument()
        self.parser.add_output_data_format("native", "csv", "ssf", default="native")
        self.parser.add_q_unit_argument()
        self.usage = self.parser.usage

    def parse_args(self, *args, **kwargs):
        """Wrapper for SASParser parse_args()"""
        return self.parser.parse_args(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        """Wrapper for SASParser add_argument()"""
        self.parser.add_argument(*args, **kwargs)
