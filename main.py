import getpass
import json
import os
import ssl
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import asyncio
import logging
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

import pandas as pd
import pymysql
from pymysql.cursors import DictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrisisResponseAgent:
    """
    Dynamic Local Crisis Response Agent powered by TiDB Serverless
    Handles multi-modal data ingestion, vector search, and automated workflows
    """
    
    def __init__(self):
        self.connection_string = None
        self.tidb_connection = None
        self.vector_store = None
        self.vector_index = None
        self.llm = None
        self.embedding_model = None
        
    def setup_ai_components(self, openai_api_key: str):
        """Initialize LLM and embedding models"""
        try:
            # Configure LlamaIndex settings
            Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
            Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)
            
            self.llm = Settings.llm
            self.embedding_model = Settings.embed_model
            
            logger.info("✅ AI components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI components: {e}")
            return False
    
    def connect_to_tidb(self, user: str = None, host: str = None, 
                       database: str = None, password: str = None):
        """Connect to TiDB Serverless with multiple SSL configuration options"""
        try:
            # Get credentials securely if not provided
            if not all([user, host, database, password]):
                print("\n🔐 TiDB Connection Setup")
                user = user or input("Enter TiDB USER: ")
                host = host or input("Enter TiDB HOST: ")
                database = database or input("Enter TiDB DATABASE name: ")
                password = password or getpass.getpass("Enter TiDB PASSWORD: ")
            
            # Try multiple connection methods in order of preference
            connection_methods = [
                self._connect_with_system_ssl,
                self._connect_with_default_ssl,
                self._connect_without_ssl
            ]
            
            for method in connection_methods:
                try:
                    success = method(user, host, database, password)
                    if success:
                        logger.info("✅ Connected to TiDB Serverless successfully!")
                        return True
                except Exception as e:
                    logger.warning(f"Connection method failed: {e}")
                    continue
            
            logger.error("❌ All connection methods failed!")
            return False
            
        except Exception as e:
            logger.error(f"❌ TiDB connection failed: {e}")
            return False
    
    def _connect_with_system_ssl(self, user, host, database, password):
        """Try connecting with system SSL certificates"""
        # Common SSL certificate locations
        ssl_paths = [
            "/etc/ssl/cert.pem",
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/usr/share/ssl/certs/ca-bundle.crt",
            "/usr/local/share/certs/ca-root-nss.crt",
            "/etc/ssl/ca-bundle.pem"
        ]
        
        # Find existing SSL certificate
        ssl_ca = None
        for path in ssl_paths:
            if os.path.exists(path):
                ssl_ca = path
                break
        
        if not ssl_ca:
            raise Exception("No system SSL certificates found")
        
        # Build connection string with SSL
        self.connection_string = (
            f"mysql+pymysql://{user}:{password}@{host}:4000/{database}"
            f"?ssl_ca={ssl_ca}&ssl_verify_cert=true&ssl_verify_identity=true"
        )
        
        # Test direct MySQL connection
        self.tidb_connection = pymysql.connect(
            host=host,
            port=4000,
            user=user,
            password=password,
            database=database,
            ssl_ca=ssl_ca,
            ssl_verify_cert=True,
            ssl_verify_identity=True,
            cursorclass=DictCursor
        )
        
        return True
    
    def _connect_with_default_ssl(self, user, host, database, password):
        """Try connecting with default SSL context"""
        # Build connection string with default SSL
        self.connection_string = (
            f"mysql+pymysql://{user}:{password}@{host}:4000/{database}"
            "?ssl_disabled=false&ssl_verify_cert=false&ssl_verify_identity=false"
        )
        
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Test direct MySQL connection
        self.tidb_connection = pymysql.connect(
            host=host,
            port=4000,
            user=user,
            password=password,
            database=database,
            ssl=ssl_context,
            cursorclass=DictCursor
        )
        
        return True
    
    def _connect_without_ssl(self, user, host, database, password):
        """Try connecting without SSL (not recommended for production)"""
        logger.warning("⚠️  Attempting connection without SSL - not recommended for production")
        
        # Build connection string without SSL
        self.connection_string = (
            f"mysql+pymysql://{user}:{password}@{host}:4000/{database}"
            "?ssl_disabled=true"
        )
        
        # Test direct MySQL connection
        self.tidb_connection = pymysql.connect(
            host=host,
            port=4000,
            user=user,
            password=password,
            database=database,
            ssl_disabled=True,
            cursorclass=DictCursor
        )
        
        return True
    
    def create_database_schema(self):
        """Create the necessary database schema and tables"""
        try:
            with self.tidb_connection.cursor() as cursor:
                # Create reports table
                create_reports_table = """
                CREATE TABLE IF NOT EXISTS reports (
                    id VARCHAR(64) PRIMARY KEY,
                    type VARCHAR(50),
                    location VARCHAR(255),
                    report_text TEXT,
                    image_url VARCHAR(500),
                    sensor_data JSON,
                    priority ENUM('high', 'medium', 'low') DEFAULT 'medium',
                    status ENUM('active', 'resolved', 'in_progress') DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_type (type),
                    INDEX idx_location (location),
                    INDEX idx_priority (priority),
                    INDEX idx_status (status),
                    INDEX idx_created_at (created_at)
                );
                """
                
                # Create incident vectors table for AI embeddings
                create_vectors_table = """
                CREATE TABLE IF NOT EXISTS incident_vectors (
                    doc_id VARCHAR(64) PRIMARY KEY,
                    vector BLOB,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES reports(id) ON DELETE CASCADE
                );
                """
                
                # Create response_actions table for tracking agent actions
                create_actions_table = """
                CREATE TABLE IF NOT EXISTS response_actions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    incident_id VARCHAR(64),
                    action_type VARCHAR(100),
                    action_data JSON,
                    status ENUM('pending', 'completed', 'failed') DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP NULL,
                    FOREIGN KEY (incident_id) REFERENCES reports(id) ON DELETE CASCADE,
                    INDEX idx_incident (incident_id),
                    INDEX idx_status (status)
                );
                """
                
                # Create notifications table
                create_notifications_table = """
                CREATE TABLE IF NOT EXISTS notifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    incident_id VARCHAR(64),
                    notification_type VARCHAR(50),
                    recipients JSON,
                    message TEXT,
                    status ENUM('sent', 'failed', 'pending') DEFAULT 'pending',
                    sent_at TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (incident_id) REFERENCES reports(id) ON DELETE CASCADE
                );
                """
                
                # Execute table creation
                for table_sql in [create_reports_table, create_vectors_table, 
                                create_actions_table, create_notifications_table]:
                    cursor.execute(table_sql)
                
                self.tidb_connection.commit()
                logger.info("✅ Database schema created successfully!")
                return True
                
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            return False
    
    def initialize_vector_store(self):
        """Initialize TiDB Vector Store for similarity search"""
        try:
            self.vector_store = TiDBVectorStore(
                connection_string=self.connection_string,
                table_name="incident_reports_vector",
                distance_strategy="cosine",
                vector_dimension=1536,  # OpenAI embeddings dimension
                drop_existing_table=False,
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            logger.info("✅ Vector store initialized successfully!")
            return storage_context
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            return None
    
    def create_sample_data(self):
        """Create sample incident data for testing"""
        sample_incidents = [
            {
                'id': 'rpt_001',
                'type': 'flood',
                'location': 'Riverside Park, cityX',
                'report_text': 'Flooding reported near Riverside Park. Water levels rising fast due to heavy rainfall.',
                'image_url': 'https://example.com/images/flood_riverside.jpg',
                'sensor_data': {"water_level_cm": 120, "rainfall_mm": 50},
                'priority': 'high',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 'rpt_002',
                'type': 'accident',
                'location': 'Oak Avenue, cityX',
                'report_text': 'Road blocked due to fallen tree at Oak Avenue. Emergency team required for clearance.',
                'image_url': 'https://example.com/images/tree_oakavenue.jpg',
                'sensor_data': {"tree_height_m": 8, "damage_level": "moderate"},
                'priority': 'medium',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'id': 'rpt_003',
                'type': 'power',
                'location': 'Midtown, cityX',
                'report_text': 'Power outage in Midtown district affecting multiple residential blocks.',
                'image_url': 'https://example.com/images/poweroutage_midtown.jpg',
                'sensor_data': {"affected_blocks": 4, "duration_min": 30},
                'priority': 'medium',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
        return sample_incidents
    
    def parse_incident_file(self, file_path: str) -> Dict[str, Any]:
        """Parse incident report file and extract structured data"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Initialize incident data
            incident = {
                'id': '',
                'type': 'incident',
                'location': 'cityX',
                'report_text': '',
                'image_url': None,
                'sensor_data': None,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'priority': 'medium'
            }
            
            # Parse structured data
            lines = content.strip().split('\n')
            text_content = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('id:'):
                    incident['id'] = line.split(':', 1)[1].strip()
                elif line.startswith('type:'):
                    incident['type'] = line.split(':', 1)[1].strip()
                elif line.startswith('location:'):
                    incident['location'] = line.split(':', 1)[1].strip()
                elif line.startswith('created_at:'):
                    incident['created_at'] = line.split(':', 1)[1].strip()
                elif line.startswith('image_url:'):
                    incident['image_url'] = line.split(':', 1)[1].strip()
                elif line.startswith('sensor_data:'):
                    sensor_json = line.split(':', 1)[1].strip()
                    try:
                        incident['sensor_data'] = json.loads(sensor_json)
                    except json.JSONDecodeError:
                        incident['sensor_data'] = {'raw': sensor_json}
                elif line and not any(line.startswith(prefix) for prefix in 
                                    ['id:', 'type:', 'location:', 'created_at:', 'image_url:', 'sensor_data:']):
                    text_content.append(line)
            
            # Combine text content
            incident['report_text'] = ' '.join(text_content)
            
            # Determine priority based on content and sensor data
            incident['priority'] = self.determine_priority(incident)
            
            return incident
            
        except Exception as e:
            logger.error(f"❌ Failed to parse file {file_path}: {e}")
            return None
    
    def determine_priority(self, incident: Dict[str, Any]) -> str:
        """Determine incident priority based on content and sensor data"""
        high_priority_keywords = [
            'fire', 'flooding', 'emergency', 'urgent', 'life', 'death', 
            'critical', 'severe', 'danger', 'immediate'
        ]
        
        text = incident.get('report_text', '').lower()
        incident_type = incident.get('type', '').lower()
        
        # Check for high priority keywords
        if any(keyword in text for keyword in high_priority_keywords):
            return 'high'
        
        # Check sensor data for critical values
        sensor_data = incident.get('sensor_data', {})
        if sensor_data:
            # Water level checks for flooding
            if 'water_level_cm' in sensor_data and sensor_data['water_level_cm'] > 100:
                return 'high'
            # Rainfall checks
            if 'rainfall_mm' in sensor_data and sensor_data['rainfall_mm'] > 40:
                return 'high'
            # Tree damage
            if 'damage_level' in sensor_data and sensor_data['damage_level'] == 'severe':
                return 'high'
        
        # Power outages affecting multiple blocks
        if 'power' in text and 'outage' in text:
            if sensor_data.get('affected_blocks', 0) > 3:
                return 'high'
            return 'medium'
        
        return 'medium'
    
    def load_and_index_documents(self, data_dir: str = "./data/reports"):
        """Load incident documents and create vector index"""
        try:
            print(f"\n📂 Loading documents from '{data_dir}'...")
            
            incidents = []
            
            # Check if directory exists and has files
            if os.path.exists(data_dir):
                # Parse incident files
                for file_path in Path(data_dir).glob("*.txt"):
                    incident = self.parse_incident_file(str(file_path))
                    if incident:
                        incidents.append(incident)
                        logger.info(f"📄 Parsed: {incident['id']} - {incident['type']}")
            
            # If no files found or directory doesn't exist, use sample data
            if not incidents:
                logger.info("📄 No incident files found, using sample data...")
                incidents = self.create_sample_data()
            
            # Insert incidents into database
            self.insert_incidents_to_db(incidents)
            
            # Create documents for vector indexing only if AI is enabled
            if self.embedding_model:
                documents = []
                for incident in incidents:
                    # Create rich document content for better embeddings
                    doc_content = f"""
                    Incident Type: {incident['type']}
                    Location: {incident['location']}
                    Description: {incident['report_text']}
                    Priority: {incident['priority']}
                    Sensor Data: {json.dumps(incident.get('sensor_data', {}), indent=2)}
                    Timestamp: {incident['created_at']}
                    """
                    
                    from llama_index.core import Document
                    doc = Document(
                        text=doc_content,
                        metadata={
                            "incident_id": incident['id'],
                            "type": incident['type'],
                            "location": incident['location'],
                            "priority": incident['priority'],
                            "timestamp": incident['created_at']
                        }
                    )
                    documents.append(doc)
                
                print(f"📊 Loaded {len(documents)} documents with enhanced metadata")
                
                # Initialize vector store
                storage_context = self.initialize_vector_store()
                if storage_context:
                    # Build vector index
                    print("🔍 Creating vector index and inserting into TiDB...")
                    self.vector_index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        show_progress=True
                    )
                    print("✅ Vector indexing complete! Ready for similarity search.")
                else:
                    print("⚠️  Vector store not available, basic functionality only")
            else:
                print("⚠️  AI components not initialized, basic functionality only")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Document loading failed: {e}")
            return False
    
    def insert_incidents_to_db(self, incidents: List[Dict[str, Any]]):
        """Insert parsed incidents into TiDB database"""
        try:
            with self.tidb_connection.cursor() as cursor:
                for incident in incidents:
                    # Prepare data for insertion
                    sql = """
                    INSERT INTO reports (id, type, location, report_text, image_url, 
                                       sensor_data, priority, status, created_at)
                    VALUES (%(id)s, %(type)s, %(location)s, %(report_text)s, %(image_url)s,
                           %(sensor_data)s, %(priority)s, 'active', %(created_at)s)
                    ON DUPLICATE KEY UPDATE
                    type=VALUES(type), location=VALUES(location), 
                    report_text=VALUES(report_text), updated_at=CURRENT_TIMESTAMP
                    """
                    
                    # Prepare parameters
                    params = {
                        'id': incident['id'],
                        'type': incident['type'],
                        'location': incident['location'],
                        'report_text': incident['report_text'],
                        'image_url': incident['image_url'],
                        'sensor_data': json.dumps(incident['sensor_data']) if incident['sensor_data'] else None,
                        'priority': incident['priority'],
                        'created_at': incident['created_at']
                    }
                    
                    cursor.execute(sql, params)
                
                self.tidb_connection.commit()
                logger.info(f"✅ Inserted {len(incidents)} incidents into database")
                
        except Exception as e:
            logger.error(f"❌ Database insertion failed: {e}")
    
    def search_similar_incidents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar incidents using vector similarity"""
        try:
            if not self.vector_index:
                logger.warning("❌ Vector index not initialized! Using basic text search...")
                return self.basic_text_search(query, top_k)
            
            # Create query engine
            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text"  # We just want the nodes, not a response
            )
            
            # Perform similarity search
            response = query_engine.query(query)
            
            similar_incidents = []
            for node in response.source_nodes:
                similar_incidents.append({
                    'incident_id': node.metadata.get('incident_id'),
                    'type': node.metadata.get('type'),
                    'location': node.metadata.get('location'),
                    'priority': node.metadata.get('priority'),
                    'similarity_score': node.score,
                    'content': node.text[:200] + "..." if len(node.text) > 200 else node.text
                })
            
            return similar_incidents
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return self.basic_text_search(query, top_k)
    
    def basic_text_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Basic text search fallback when vector search is not available"""
        try:
            with self.tidb_connection.cursor() as cursor:
                # Simple text search using LIKE
                cursor.execute("""
                    SELECT id, type, location, priority, report_text, created_at
                    FROM reports 
                    WHERE report_text LIKE %s OR type LIKE %s OR location LIKE %s
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (f"%{query}%", f"%{query}%", f"%{query}%", top_k))
                
                results = cursor.fetchall()
                
                similar_incidents = []
                for result in results:
                    similar_incidents.append({
                        'incident_id': result['id'],
                        'type': result['type'],
                        'location': result['location'],
                        'priority': result['priority'],
                        'similarity_score': 0.5,  # Placeholder score
                        'content': result['report_text'][:200] + "..." if len(result['report_text']) > 200 else result['report_text']
                    })
                
                return similar_incidents
                
        except Exception as e:
            logger.error(f"❌ Basic text search failed: {e}")
            return []
    
    def analyze_incident_with_ai(self, incident_id: str) -> Dict[str, Any]:
        """Analyze incident using LLM and generate response recommendations"""
        try:
            # Get incident details from database
            incident = self.get_incident_by_id(incident_id)
            if not incident:
                return {"error": "Incident not found"}
            
            # Find similar past incidents
            similar_incidents = self.search_similar_incidents(
                f"{incident['type']} {incident['location']} {incident['report_text']}"
            )
            
            # Prepare analysis prompt
            analysis_prompt = f"""
            You are an AI crisis response analyst. Analyze this incident and provide actionable recommendations:

            CURRENT INCIDENT:
            ID: {incident['id']}
            Type: {incident['type']}
            Location: {incident['location']}
            Description: {incident['report_text']}
            Priority: {incident['priority']}
            Sensor Data: {json.dumps(incident.get('sensor_data'), indent=2)}

            SIMILAR PAST INCIDENTS:
            {json.dumps(similar_incidents, indent=2)}

            Provide analysis in this JSON format:
            {{
                "severity_assessment": "string",
                "immediate_actions": ["action1", "action2"],
                "resource_requirements": ["resource1", "resource2"],
                "estimated_response_time": "minutes",
                "evacuation_needed": boolean,
                "public_alert_message": "string",
                "coordination_steps": ["step1", "step2"]
            }}
            """
            
            # Get AI analysis
            if self.llm:
                response = self.llm.complete(analysis_prompt)
                
                try:
                    # Try to parse as JSON
                    analysis = json.loads(response.text)
                except json.JSONDecodeError:
                    # Fallback to text response
                    analysis = {
                        "raw_analysis": response.text,
                        "similar_incidents": similar_incidents
                    }
            else:
                # Fallback analysis without LLM
                analysis = self.generate_fallback_analysis(incident, similar_incidents)
            
            # Log analysis action
            self.log_response_action(incident_id, "ai_analysis", analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return {"error": str(e)}
    
    def generate_fallback_analysis(self, incident: Dict, similar_incidents: List) -> Dict:
        """Generate basic analysis without LLM"""
        priority_map = {'high': 'CRITICAL', 'medium': 'MODERATE', 'low': 'LOW'}
        
        return {
            "severity_assessment": f"{priority_map.get(incident['priority'], 'UNKNOWN')} - {incident['type'].upper()}",
            "immediate_actions": self.get_standard_actions(incident['type']),
            "resource_requirements": self.get_standard_resources(incident['type']),
            "estimated_response_time": "5-15",
            "evacuation_needed": incident['priority'] == 'high',
            "public_alert_message": f"{incident['type'].title()} reported at {incident['location']}. Please avoid the area.",
            "coordination_steps": ["Notify emergency services", "Secure perimeter", "Assess damage"],
            "similar_incidents_found": len(similar_incidents)
        }
    
    def get_standard_actions(self, incident_type: str) -> List[str]:
        """Get standard response actions by incident type"""
        actions = {
            'fire': ["Dispatch fire department", "Establish safety perimeter", "Evacuate nearby buildings"],
            'flood': ["Deploy sandbags", "Close affected roads", "Open emergency shelters"],
            'power': ["Contact utility company", "Deploy backup generators", "Check critical facilities"],
            'accident': ["Dispatch EMS", "Clear traffic", "Investigate cause"],
            'medical': ["Dispatch ambulance", "Clear access routes", "Prepare receiving hospital"],
            'weather': ["Issue weather alert", "Open shelters", "Secure outdoor equipment"]
        }
        return actions.get(incident_type, ["Assess situation", "Deploy appropriate resources"])
    
    def get_standard_resources(self, incident_type: str) -> List[str]:
        """Get standard resource requirements by incident type"""
        resources = {
            'fire': ["Fire trucks", "EMT units", "Police backup"],
            'flood': ["Sandbags", "Water pumps", "Rescue boats"],
            'power': ["Utility crews", "Backup generators", "Emergency lighting"],
            'accident': ["Ambulance", "Tow trucks", "Traffic control"],
            'medical': ["Ambulance", "Paramedics", "Medical helicopter"],
            'weather': ["Emergency supplies", "Shelter materials", "Communication equipment"]
        }
        return resources.get(incident_type, ["General emergency response team"])
    
    def get_incident_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve incident details from database"""
        try:
            with self.tidb_connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM reports WHERE id = %s", 
                    (incident_id,)
                )
                result = cursor.fetchone()
                
                if result and result.get('sensor_data'):
                    try:
                        result['sensor_data'] = json.loads(result['sensor_data'])
                    except json.JSONDecodeError:
                        pass
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to get incident {incident_id}: {e}")
            return None
    
    def log_response_action(self, incident_id: str, action_type: str, action_data: Dict):
        """Log response action to database"""
        try:
            with self.tidb_connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO response_actions (incident_id, action_type, action_data, status)
                       VALUES (%s, %s, %s, 'completed')""",
                    (incident_id, action_type, json.dumps(action_data))
                )
                self.tidb_connection.commit()
        except Exception as e:
            logger.error(f"❌ Failed to log action: {e}")
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get real-time dashboard statistics"""
        try:
            with self.tidb_connection.cursor() as cursor:
                stats = {}
                
                # Active incidents
                cursor.execute("SELECT COUNT(*) as count FROM reports WHERE status = 'active'")
                stats['active_incidents'] = cursor.fetchone()['count']
                
                # Resolved today
                cursor.execute("""
                    SELECT COUNT(*) as count FROM reports 
                    WHERE status = 'resolved' AND DATE(created_at) = CURDATE()
                """)
                stats['resolved_today'] = cursor.fetchone()['count']
                
                # High priority incidents
                cursor.execute("""
                    SELECT COUNT(*) as count FROM reports 
                    WHERE priority = 'high' AND status = 'active'
                """)
                stats['high_priority'] = cursor.fetchone()['count']
                
                # Recent incidents
                cursor.execute("""
                    SELECT id, type, location, priority, created_at, report_text
                    FROM reports 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                stats['recent_incidents'] = cursor.fetchall()
                
                return stats
        
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}


def main():
    """Main execution function"""
    print("🚨 Dynamic Local Crisis Response Agent - TiDB AgentX")
    print("=" * 60)
    
    # Initialize agent
    agent = CrisisResponseAgent()
    
    # Get OpenAI API key for AI features
    print("\n🤖 AI Integration Setup (Optional)")
    print("Enter OpenAI API Key to enable advanced AI features, or press Enter to skip")
    openai_key = getpass.getpass("OpenAI API Key: ").strip()
    
    if openai_key:
        if agent.setup_ai_components(openai_key):
            print("✅ AI components initialized - Advanced features enabled")
        else:
            print("⚠️  AI initialization failed - Basic features only")
    else:
        print("ℹ️  Skipping AI setup - Basic features enabled")
    
    # Connect to TiDB
    print("\n📊 Connecting to TiDB Serverless...")
    if not agent.connect_to_tidb():
        print("❌ Failed to connect to TiDB. Please check your credentials and try again.")
        print("\n💡 Common solutions:")
        print("1. Verify your TiDB credentials are correct")
        print("2. Check your internet connection")
        print("3. Ensure TiDB cluster is running")
        return
    
    # Create database schema
    print("\n🏗️  Setting up database schema...")
    if not agent.create_database_schema():
        print("❌ Failed to create database schema.")
        return
    
    # Load and index documents
    print("\n📚 Loading incident data...")
    data_directory = input("Enter data directory path (default: ./data/reports): ").strip()
    if not data_directory:
        data_directory = "./data/reports"
    
    if not agent.load_and_index_documents(data_directory):
        print("❌ Failed to load documents.")
        return
    
    print("\n🎯 Crisis Response System Ready!")
    print("\nAvailable commands:")
    print("  'stats' - Show dashboard statistics")
    print("  'analyze <incident_id>' - Analyze specific incident (e.g., analyze rpt_001)")
    print("  'search <query>' - Search similar incidents (e.g., search flood)")
    print("  'list' - List all incidents")
    print("  'help' - Show this help message")
    print("  'quit' - Exit system")
    
    # Interactive loop
    while True:
        try:
            command = input("\n🔧 > ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
                
            elif command == 'help':
                print("\nAvailable commands:")
                print("  'stats' - Show dashboard statistics")
                print("  'analyze <incident_id>' - Analyze specific incident")
                print("  'search <query>' - Search similar incidents")
                print("  'list' - List all incidents")
                print("  'help' - Show this help message")
                print("  'quit' - Exit system")
                
            elif command == 'stats':
                stats = agent.get_dashboard_stats()
                print(f"\n📊 Dashboard Statistics:")
                print(f"Active Incidents: {stats.get('active_incidents', 0)}")
                print(f"Resolved Today: {stats.get('resolved_today', 0)}")
                print(f"High Priority: {stats.get('high_priority', 0)}")
                
                recent = stats.get('recent_incidents', [])
                if recent:
                    print(f"\n📋 Recent Incidents ({len(recent)}):")
                    for incident in recent[:5]:
                        print(f"  • {incident['id']}: {incident['type']} - {incident['location']} [{incident['priority']}]")
                
            elif command == 'list':
                stats = agent.get_dashboard_stats()
                recent = stats.get('recent_incidents', [])
                if recent:
                    print(f"\n📋 All Incidents ({len(recent)}):")
                    for incident in recent:
                        print(f"  • {incident['id']}: {incident['type']} - {incident['location']} [{incident['priority']}]")
                        print(f"    {incident['report_text'][:100]}{'...' if len(incident['report_text']) > 100 else ''}")
                        print()
                else:
                    print("No incidents found.")
                
            elif command.startswith('analyze '):
                parts = command.split(' ', 1)
                if len(parts) < 2:
                    print("Usage: analyze <incident_id>")
                    continue
                    
                incident_id = parts[1].strip()
                print(f"\n🤖 Analyzing incident {incident_id}...")
                analysis = agent.analyze_incident_with_ai(incident_id)
                
                if "error" in analysis:
                    print(f"❌ {analysis['error']}")
                else:
                    print(f"\n🔍 AI Analysis for {incident_id}:")
                    print(json.dumps(analysis, indent=2))
                
            elif command.startswith('search '):
                parts = command.split(' ', 1)
                if len(parts) < 2:
                    print("Usage: search <query>")
                    continue
                    
                query = parts[1].strip()
                print(f"\n🔍 Searching for incidents similar to '{query}'...")
                results = agent.search_similar_incidents(query)
                
                if results:
                    print(f"\nFound {len(results)} similar incidents:")
                    for i, result in enumerate(results, 1):
                        score_display = f"({result['similarity_score']:.3f})" if result['similarity_score'] != 0.5 else "(basic match)"
                        print(f"  {i}. {result['incident_id']}: {result['type']} {score_display}")
                        print(f"     Location: {result['location']} | Priority: {result['priority']}")
                        print(f"     {result['content'][:150]}{'...' if len(result['content']) > 150 else ''}")
                        print()
                else:
                    print("No similar incidents found.")
                    
            else:
                print("❓ Unknown command. Type 'help' for available commands or 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Crisis Response Agent shutting down...")
    if agent.tidb_connection:
        agent.tidb_connection.close()
        print("✅ Database connection closed")


if __name__ == "__main__":
    main()