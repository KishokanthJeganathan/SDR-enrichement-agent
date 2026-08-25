
import argparse

from .agent_simple import run
from .models import Enquiry
from .render import render_brief


def main() -> None:
    parser = argparse.ArgumentParser(description="SDR account brief agent")
    parser.add_argument("--company", required=True, help="Company name to research")
    parser.add_argument("--email", required=True, help="Contact's email (domain is used)")
    parser.add_argument("--name", default="Unknown", help="Contact's name (passed through only)")
    parser.add_argument("--message", default="", help="Free-text enquiry message")
    args = parser.parse_args()

    enquiry = Enquiry(
        contact_name=args.name,
        contact_email=args.email,
        company_name=args.company,
        message=args.message,
    )
    trace = run(enquiry)
    print(render_brief(trace.brief))


if __name__ == "__main__":
    main()
