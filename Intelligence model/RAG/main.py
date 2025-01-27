import asyncio
import os
import torch
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import sounddevice as sd
import soundfile as sf
import numpy as np
from TTS.api import TTS
from datetime import datetime
from pymongo import MongoClient
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configuration
DB_FAISS_PATH = r'C:\Users\sharo\Desktop\Ai-character-generation\Intelligence model\RAG\vectorstores\db_faiss'
SAMPLE_RATE = 16000
RECORD_SECONDS = 10

class ConversationMemory:
    def __init__(self, mongo_uri):
        self.client = MongoClient(mongo_uri)
        self.db = self.client['conversation_history']
        self.conversations = self.db['conversations']
        self.current_topic = None
        
        # Initialize emotion analysis pipeline
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    
    def analyze_emotion(self, text):
        """Analyze the emotional content of text"""
        emotions = self.emotion_classifier(text)[0]
        emotion_dict = {item['label']: item['score'] for item in emotions}
        dominant_emotion = max(emotions, key=lambda x: x['score'])
        
        return {
            "mood": dominant_emotion['label'],
            "emotions": emotion_dict
        }
    
    def store_interaction(self, query, response, session_id):
        """Store a conversation interaction with timestamp and emotions"""
        timestamp = datetime.utcnow()
        emotional_state = self.analyze_emotion(query)
        
        interaction = {
            "session_id": session_id,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "emotional_state": emotional_state
        }
        
        self.conversations.insert_one(interaction)
    
    def get_recent_interactions(self, session_id, limit=5):
        """Retrieve recent interactions for context"""
        recent = self.conversations.find(
            {"session_id": session_id},
            sort=[("timestamp", -1)],
            limit=limit
        )
        return list(recent)
    
    def update_current_topic(self, query, is_followup=False):
        """Update the current conversation topic"""
        if not is_followup:
            self.current_topic = query
    
    def get_current_topic(self):
        """Get the current conversation topic"""
        return self.current_topic
    
    def get_previous_response(self, session_id):
        """Get the most recent response for the current topic"""
        if self.current_topic:
            recent = self.conversations.find_one(
                {
                    "session_id": session_id,
                    "query": self.current_topic
                }
            )
            return recent['response'] if recent else None
        return None

class SpeechRecognitionBot:
    def __init__(self):
        # Force GPU usage
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available. This script requires a CUDA-compatible GPU.")
        
        torch.cuda.set_device(0)  # Set to first GPU if multiple are available
        
        # Initialize Whisper model with forced GPU
        self.whisper_model = WhisperModel(
            model_size_or_path='base',
            device='cuda', 
            compute_type='float16'
        )   
        
        # Initialize TTS model
        self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Initialize QA components
        self.qa_chain = None
        
        # Initialize conversation memory
        mongo_uri = "mongodb+srv://hahashinchan19:4XuzNOXvRjYcishs@cluster0.xuczb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        self.memory = ConversationMemory(mongo_uri)
        
        # Generate a unique session ID
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Initialize followup phrases
        self.followup_phrases = [
            "tell me more about it",
            "tell me more",
            "what else",
            "continue",
            "go on",
            "and then",
            "what happened next",
            "can you elaborate",
            "please elaborate",
            "could you explain more"
        ]
    
    def record_audio(self, duration=RECORD_SECONDS):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype=np.float32
        )
        sd.wait()
        return recording
    
    def save_audio(self, audio_data, filename='query_audio.wav'):
        """Save recorded audio to WAV file"""
        sf.write(filename, audio_data, SAMPLE_RATE)
        return filename
    
    def play_audio(self, filename):
        """Play audio response"""
        data, fs = sf.read(filename)
        sd.play(data, fs)
        sd.wait()
    
    def generate_voice_response(self, text, output_path='response.wav'):
        """Generate voice response using XTTS-v2"""
        reference_wav = r"C:\Users\sharo\Desktop\Ai-character-generation\Intelligence model\RAG\demo.wav"
        
        self.tts_model.tts_to_file(
            text=text, 
            file_path=output_path,
            speaker_wav=reference_wav,
            language="en"
        )
        return output_path
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio using Faster Whisper"""
        try:
            segments, info = self.whisper_model.transcribe(
                audio_file, 
                beam_size=5,
                language='en'
            )
            
            transcription = ' '.join([segment.text for segment in segments])
            return transcription.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    async def initialize_qa_bot(self):
        """Initialize QA components"""
        hugging_face_embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-V2', 
            model_kwargs={'device': 'cpu'}
        )

        faiss_db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings=hugging_face_embeddings, 
            allow_dangerous_deserialization=True
        )
    
        # Initialize Groq LLM with Llama3 model
        groq_model = ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv('GROQ_API_KEY'),
            temperature=0.5,
            max_tokens=2000
        )

        qa_prompt = PromptTemplate(
            template="""
            You're tasked with providing a helpful response based on the given context and question.
            Accuracy is paramount, so if you're uncertain, it's best to acknowledge that rather than providing potentially incorrect information.

            Context: {context}
            Question: {question}

            Please craft a clear and informative response that directly addresses the question.
            """,
            input_variables=['context', 'question']
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=groq_model,
            chain_type='stuff',
            retriever=faiss_db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': qa_prompt}
        )
    
    async def generate_response(self, query):
        """Generate response using Gemini API with conversation history"""
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            return "Error: Gemini API key not found"
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Get recent conversation history
        recent_interactions = self.memory.get_recent_interactions(self.session_id)
        conversation_context = "\n".join([
            f"User ({interaction['timestamp']}): {interaction['query']}\n"
            f"Assistant: {interaction['response']}"
            for interaction in recent_interactions
        ])
        
        # Check if this is a follow-up question
        is_followup = any(phrase in query.lower() for phrase in self.followup_phrases)
        
        if is_followup and self.memory.get_current_topic():
            previous_topic = self.memory.get_current_topic()
            previous_response = self.memory.get_previous_response(self.session_id)
            
            followup_prompt = f"""
            Generate a follow-up response about: "{previous_topic}"
            
            Previous response given: "{previous_response}"
            
            Requirements:
            - Provide NEW information not mentioned in the previous response
            - Continue naturally from the previous context
            - Focus on different aspects or details
            - Maintain the conversational style with fillers
            - Show appropriate expressions in [ ]
            - Consider user's emotional state: {self.memory.analyze_emotion(query)['mood']}
            - Be engaging and informative
            
            Do not repeat information from the previous response.
            """
            
            try:
                synthesis_response = await model.generate_content_async(followup_prompt)
                response = synthesis_response.text.strip()
                
                # Store the interaction with same topic
                self.memory.store_interaction(query, response, self.session_id)
                return response
                
            except Exception as e:
                return f"Error generating follow-up response: {str(e)}"
        
        # Check if the query is a simple conversational phrase
        conversational_phrases = [
            "thank you", "thanks", "okay", "bye", "goodbye", "see you",
            "good morning", "good afternoon", "good evening", "hello", "hi"
        ]
        
        is_conversational = any(phrase in query.lower() for phrase in conversational_phrases)
        
        if is_conversational:
            conversation_prompt = f"""
            Generate a natural, conversational response to: "{query}"
            
            Consider:
            - User's Emotional State: {self.memory.analyze_emotion(query)['mood']}
            - Keep it brief and natural
            - Use conversational fillers naturally
            - Show appropriate expressions in [ ]
            - Match the user's tone
            """
            
            try:
                synthesis_response = await model.generate_content_async(conversation_prompt)
                response = synthesis_response.text.strip()
                
                # Store the interaction in memory
                self.memory.store_interaction(query, response, self.session_id)
                return response
                
            except Exception as e:
                return f"Error generating response: {str(e)}"
        
        # For historical queries, proceed with the original logic
        try:
            res = await self.qa_chain.acall(query)
            llama_answer = res["result"]
            
            synthesis_prompt = f"""
            You are an expert Historical assistant providing a precise answer.

            Previous Conversation Context:
            {conversation_context}

            Context from initial research:
            {llama_answer}

            Original Query: {query}

            User's Emotional State: {self.memory.analyze_emotion(query)['mood']}

            Task: 
            - Provide a direct, concise, and informative response
            - Consider the user's emotional state and previous conversation context
            - Precisely address the specific History query
            - Use the context to enhance your answer
            - Do not include unnecessary information
            - Keep the response length matches the depth of the original query
            - You should sound like a human, use words like uhm-uhm, you know, etc.
            - Try to make it sound like a conversation
            - use fillers to make the response sound natural
            - show expressions like anger, happy or laughter based on the contexts
            - if showing expression make sure to quote the expresion in [ ]
            - Be friendly and helpful
            - Maintain conversation continuity with previous exchanges
            """
            
            synthesis_response = await model.generate_content_async(synthesis_prompt)
            response = synthesis_response.text.strip()
            
            # Update current topic and store the interaction
            self.memory.update_current_topic(query)
            self.memory.store_interaction(query, response, self.session_id)
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def run(self):
        """Main interaction loop"""
        # Initialize QA components
        await self.initialize_qa_bot()
        
        print("History Voice Assistant")
        print("Speak your history-related query. Say 'exit' to quit.")
        print(f"Session ID: {self.session_id}")
        
        while True:
            try:
                # Record audio
                audio_data = self.record_audio()
                audio_file = self.save_audio(audio_data)
                
                # Transcribe query
                query = self.transcribe_audio(audio_file)
                
                if not query:
                    print("Could not transcribe. Please try again.")
                    continue
                
                print(f"Transcribed Query: {query}")
                
                # Check for exit
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                # Generate text response
                response = await self.generate_response(query)
                print("\nBot's Response:", response)
                
                # Generate and play voice response
                voice_response_file = self.generate_voice_response(response)
                self.play_audio(voice_response_file)
                
                print("\nReady for your next query.")
                print("-"*50 + "\n")
            
            except Exception as e:
                print(f"An error occurred: {e}")

async def main():
    bot = SpeechRecognitionBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())