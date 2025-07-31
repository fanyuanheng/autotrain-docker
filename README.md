# AutoTrain Docker

A simple Docker setup to quickly spin up AutoTrain Advanced in a local Docker environment for machine learning model training and fine-tuning.

## Overview

This project provides a containerized version of [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) using Docker Compose, making it easy to run AutoTrain locally without complex setup requirements.

## Features

- 🐳 **Docker-based**: No need to install AutoTrain dependencies locally
- 🚀 **Quick Setup**: Get started with just a few commands
- 📁 **Persistent Data**: Your training data and models are preserved in the `autotrain-data` directory
- 🔐 **Secure**: Hugging Face token stored in environment variables
- 🌐 **Web Interface**: Access AutoTrain UI at `http://localhost:7860`

## Prerequisites

- Docker and Docker Compose installed on your system
- A Hugging Face account and access token

## Setup

1. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd autotrain-docker
   ```

2. **Create the required data directory:**
   ```bash
   # Create autotrain-data directory for volume mounting
   mkdir -p autotrain-data
   ```

3. **Create a `.env` file in the project root:**
   ```bash
   # Create .env file
   touch .env
   ```

4. **Add your Hugging Face token to the `.env` file:**
   ```bash
   # Add your HF token to .env file
   echo "HF_TOKEN=your_huggingface_token_here" >> .env
   ```

   **Important:** Replace `your_huggingface_token_here` with your actual Hugging Face token. You can get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

## Usage

### Starting AutoTrain

1. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```

2. **Access the AutoTrain web interface:**
   Open your browser and navigate to `http://localhost:7860`

### Stopping AutoTrain

```bash
docker-compose down
```

### Running in Background

```bash
docker-compose up -d --build
```

### Viewing Logs

```bash
docker-compose logs -f autotrain
```

## Project Structure

```
autotrain-docker/
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile           # Docker image definition
├── .env                 # Environment variables (create this)
├── autotrain-data/      # Persistent data directory (create this)
└── README.md           # This file
```

## Configuration

### Port Configuration

The AutoTrain web interface runs on port `7860` by default. You can change this in `docker-compose.yml`:

```yaml
ports:
  - "YOUR_PORT:7860"
```

### Data Persistence

Your training data, models, and AutoTrain outputs are stored in the `autotrain-data` directory, which is mounted to `/app/data` inside the container.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | Your Hugging Face access token | Yes |

## Troubleshooting

### Container Won't Start

1. Check if Docker is running
2. Verify your `.env` file exists and contains a valid `HF_TOKEN`
3. Check logs: `docker-compose logs autotrain`

### Port Already in Use

If port 7860 is already in use, change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "7861:7860"  # Use port 7861 instead
```

### Permission Issues

If you encounter permission issues with the data directory:

```bash
sudo chown -R $USER:$USER autotrain-data/
```

## Getting Your Hugging Face Token

1. Go to [Hugging Face](https://huggingface.co/)
2. Sign in to your account
3. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Create a new token with appropriate permissions
5. Copy the token and add it to your `.env` file

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 