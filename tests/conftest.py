"""Keep the offline tests out of LangSmith: the package loads .env,
which turns tracing on when a LangSmith key is present."""

import os

os.environ["LANGSMITH_TRACING"] = "false"
