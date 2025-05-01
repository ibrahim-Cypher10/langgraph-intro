# LangGraph Intro

This project gives a fundamental introduction to LangGraph by using it to build a simple but powerful data analytics AI agent that can query your database, perform analyses, and generate visualizations. This is an end-to-end, full-deployed AI agent that will teach you core Langgraph concepts so that you can build amazing AI systems yourself. It's meant to be a starting point so add to this example to expand on the agent's capabilities and create your own full-deployed, powerful AI agents. Everything covered is free except for usage of the OpenAI API - however feel free to adopt the code to use any provider including local models for free.

Watch the YT video here and **Subscribe** to my free community to get the accompanying cheat-sheet with in-depth notes and more advanced topics and diagrams.

## Requirements

- Python 3.12
- Docker
- Supabase account (optional for persistance and deployment)
- Langsmith account (optional for tracing)
- Render account (optional for deployment)

## Project Structure

- `onlyvans`: The package containing the agent definition.
- `sample_data`: The sample data we'll load into our postgresql database.
- `chat.py`: A simple test chat to test the agent in the local environment.
- `app.py`: A simple command-line "frontend" for testing the deployed agent.
- `langgraph.json`: The LangGraph configuration file where we specify the agent graph to expose via the API.
- `.env.sample`: The environment variables required to run the project. This should be copied to `.env` and filled in with your own secrets.

## Setup

Only steps 1 through 3 are required to get the agent up and running. Steps 4-7 are optional if you want to build a fully-deployed agent.

### 1. Clone the repo

Navigate to the directory where you want to clone the repo.

Clone it to your local machine.

```bash
git clone https://github.com/onlyvans/langgraph-intro.git
```

Enter the project directory.

```bash
cd langgraph-intro
```

### 2. Install packages in a virtual environment

#### Using uv (recommended)

The recommended way is to use the package manager uv as it's fast, efficient, and makes the whole process much easier! See [uv](https://github.com/onlyvans/uv) for information on how to install uv.

If using uv, we can create a virtual environment in the project directory and install the required packages with one command.

```bash
uv sync
```

We also want to install the local onlyvans package in editable mode so that we can import it to our frontend scripts, and so that any changes you make will be automatically reflected.

```bash
uv install -e .
```

### 3. Set up the database (optional)

We are using Supabase for our Postgresql database. Supabase is easy to set up and offers a generous free tier. You can optionally connect any database of your choosing by simply changing the SUPABASE_URL and DATABASE_URI in the .env file.

1. Create a Supabase account at supabase.com.
2. Create a new project.
3. Select the new project and navigate to the Table Editor in the side menu.
4. Create a new schema called `onlyvans`.
5. Make sure the `onlyvans` schema is selected.
6. Create a new table called `creators`.
![alt text](static/import_csv.png)
7. You'll see a pop up and you can select the import data from CSV button to automatically load the data and set the schema. Repeat this for the other tables by loading all of the data in the sample_data folder.

### 4. Create a Langsmith account (optional)

Langsmith is used for logging and monitoring. This is completely optional but highly recommended as it makes it easy to track and debug your LangGraph applications. You can create a free account at [langsmith.com](https://langsmith.com/).

### 5. Create a Render account (optional)

Render is used for deploying the LangGraph server. This is completely optional but highly recommended as it makes it easy to deploy your LangGraph applications. You can create a free account at [render.com](https://render.com/).

### 6. Set up the .env file

1. Copy the `.env.sample` file to `.env`.
2. Fill in the environment variables with your own secrets.

## Deployment

### 1. Build the langgraph image

```bash
langgraph build -t langgraph-intro
```

### 2. Run the langgraph server in docker

docker run --env-file .env -p 8123:8000 langgraph-intro
