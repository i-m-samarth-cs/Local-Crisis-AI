# Dynamic Local Crisis Response AI

## Overview
Dynamic Local Crisis Response AI is an intelligent, multi-step agent that automates real-time emergency management workflows using TiDB Serverless vector search and advanced AI techniques. It ingests multi-modal reports including text, images, and sensor data, indexes them for rapid retrieval, chains LLM analysis for incident prioritization, and automatically triggers notifications and response coordination.

## Features
- Multi-modal data ingestion (text, images, sensor feeds)
- Vector search indexing with TiDB Serverless for high scalability
- LLM-powered multi-step workflows for real-time summarization and decision-making
- Integration with external APIs for alerting, mapping, and responder coordination
- Batch insertion for efficient handling of large data volumes

## Installation & Setup
1. Sign up for a free TiDB Cloud Serverless account and create a cluster.
2. Clone this repository.
3. Prepare data files in the `data/reports` directory (sample files provided).
4. Install dependencies:
pip install llama-index llama-index-vector-stores-tidbvector pymysql

text
5. Edit the connection string in the main Python script with your TiDB credentials.

## Usage
Run the ingestion and indexing script:
python tidb_ingest_index.py

text
The script will load incident reports, embed them, and store vectors in TiDB Serverless.

## Challenges
- Designing a unified schema for multi-modal data ingestion.
- Managing large batch inserts to avoid packet size limitations.
- Orchestrating multi-step agent workflows combining search, LLM, and external API calls.
- Ensuring secure and reliable connectivity with TiDB Cloud.

## Accomplishments
- Built an end-to-end AI agent pipeline that automatically indexes and processes crisis reports.
- Implemented scalable vector storage and search on TiDB Serverless.
- Achieved smooth integration of LLM reasoning with real-time alerting workflows.
- Created reusable sample data and comprehensive documentation for ease of adoption.

## Future Work
- Expand live IoT and citizen report integrations.
- Enhance decision logic with region- and situation-specific models.
- Add multi-language support and accessibility features.
- Open-source the platform for wider community use and contribution.

## License
MIT License
