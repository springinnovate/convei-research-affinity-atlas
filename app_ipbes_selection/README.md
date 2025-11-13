# app_ipbes_selection

This application is intended for filtering and selecting potential IPBES authors, supporting workflows where candidate information is processed, screened, or ranked.

**Current Status:**
The code and data in this application are **unstable**, **partially complete**, and their state is **unknown**.
It should be considered experimental and **not relied upon** for production or reliable analysis.

## Environment Requirements

A `.env` file must exist in the project root with the following variables defined:

- `OPEN_API_KEY` – API key for accessing the OpenAI API
- `DATABASE_URL` – connection string for the application’s database

The application will not run correctly without these environment variables.
