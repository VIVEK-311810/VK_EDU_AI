import os
import tempfile
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
WHISPER_MODEL_SIZE = os.getenv('WHISPER_MODEL_SIZE', 'small')  # Options: tiny, base, small, medium, large
MAX_AUDIO_SIZE_MB = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model variable (loaded once at startup)
model = None

def load_whisper_model():
    """Load Whisper model at startup"""
    global model
    logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on device: {DEVICE}")
    try:
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
        logger.info(f"Whisper model loaded successfully on {DEVICE}")
        
        # Log GPU info if available
        if DEVICE == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': WHISPER_MODEL_SIZE,
        'device': DEVICE,
        'model_loaded': model is not None,
        'cuda_available': torch.cuda.is_available()
    }), 200

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio file sent via multipart/form-data
    
    Expected form fields:
    - audio: audio file (required)
    - session_id: session identifier (optional)
    - language: target language code (optional, default: auto-detect)
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            logger.warning("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id', 'unknown')
        language = request.form.get('language', None)  # None = auto-detect
        
        if audio_file.filename == '':
            logger.warning("Empty filename in request")
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Check file size
        audio_file.seek(0, os.SEEK_END)
        file_size = audio_file.tell()
        audio_file.seek(0)
        
        if file_size > MAX_AUDIO_SIZE_MB * 1024 * 1024:
            logger.warning(f"File too large: {file_size / 1024 / 1024:.2f} MB")
            return jsonify({'error': f'File too large. Maximum size: {MAX_AUDIO_SIZE_MB}MB'}), 400
        
        logger.info(f"Received audio file for session '{session_id}': {audio_file.filename} ({file_size / 1024:.2f} KB)")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)
        
        try:
            # Transcribe audio
            logger.info(f"Transcribing audio for session '{session_id}'...")
            
            transcribe_options = {
                'fp16': DEVICE == 'cuda',  # Use FP16 only on CUDA
                'language': language,
                'task': 'transcribe'
            }
            
            result = model.transcribe(temp_path, **transcribe_options)
            
            transcript = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            
            logger.info(f"Transcription complete for session '{session_id}': {len(transcript)} characters, language: {detected_language}")
            
            # Return response
            return jsonify({
                'success': True,
                'transcript': transcript,
                'text': transcript,  # Alias for compatibility
                'session_id': session_id,
                'detected_language': detected_language,
                'segments': len(result.get('segments', [])),
                'duration': result.get('duration', 0)
            }), 200
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Transcription failed',
            'details': str(e)
        }), 500

@app.route('/transcribe-stream', methods=['POST'])
def transcribe_audio_stream():
    """
    Transcribe raw audio data without file I/O (low-latency streaming)

    Expected JSON:
    {
        "audio_data": [0.1, -0.05, 0.2, ...],  # Float32 PCM samples
        "sample_rate": 16000,                   # Sample rate (must be 16000 for Whisper)
        "session_id": "session123",             # Session identifier
        "language": "en"                        # Optional language hint
    }
    """
    try:
        # Parse JSON request
        data = request.get_json()

        if not data or 'audio_data' not in data:
            logger.warning("No audio_data in stream request")
            return jsonify({'error': 'No audio_data provided'}), 400

        audio_data = data.get('audio_data')
        session_id = data.get('session_id', 'unknown')
        sample_rate = data.get('sample_rate', 16000)
        language = data.get('language', None)

        if not isinstance(audio_data, list) or len(audio_data) == 0:
            logger.warning("Invalid audio_data format")
            return jsonify({'error': 'audio_data must be a non-empty array'}), 400

        logger.info(f"Received raw audio stream for session '{session_id}': {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f} seconds)")

        # Convert to numpy float32 array
        # Browser sends Float32 PCM in range [-1, 1], which is exactly what Whisper expects
        audio_np = np.array(audio_data, dtype=np.float32)

        # Validate sample rate (Whisper requires 16kHz)
        if sample_rate != 16000:
            logger.warning(f"Sample rate {sample_rate} != 16000, audio quality may be affected")

        # Transcribe directly - Whisper accepts numpy arrays!
        # This bypasses ffmpeg entirely, eliminating file I/O and parsing overhead
        logger.info(f"Transcribing raw audio stream for session '{session_id}'...")

        transcribe_options = {
            'fp16': DEVICE == 'cuda',  # Use FP16 only on CUDA
            'language': language,
            'task': 'transcribe'
        }

        # Direct transcription from numpy array - NO file I/O!
        result = model.transcribe(audio_np, **transcribe_options)

        transcript = result['text'].strip()
        detected_language = result.get('language', 'unknown')

        logger.info(f"Stream transcription complete for session '{session_id}': {len(transcript)} characters, language: {detected_language}")

        # Return response
        return jsonify({
            'success': True,
            'transcript': transcript,
            'text': transcript,  # Alias for compatibility
            'session_id': session_id,
            'detected_language': detected_language,
            'audio_duration': len(audio_data) / sample_rate,
            'sample_count': len(audio_data)
        }), 200

    except Exception as e:
        logger.error(f"Error during stream transcription: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Stream transcription failed',
            'details': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_size': WHISPER_MODEL_SIZE,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'model_loaded': model is not None
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit exceeded"""
    return jsonify({'error': f'File too large. Maximum size: {MAX_AUDIO_SIZE_MB}MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model at startup
    load_whisper_model()
    
    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5009))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting transcription server on {host}:{port}")
    logger.info(f"Using Whisper model: {WHISPER_MODEL_SIZE}")
    logger.info(f"Device: {DEVICE}")
    
    # Configure Flask app
    app.config['MAX_CONTENT_LENGTH'] = MAX_AUDIO_SIZE_MB * 1024 * 1024
    
    # Run server
    app.run(
        host=host,
        port=5000,
        debug=True,
        threaded=True  # Enable threading for concurrent requests
    )
