import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:typed_data';

final FlutterTts flutterTts = FlutterTts();
// IMPORTANT: Do not expose API keys in your code. Use environment variables.
var _apikey = "AIzaSyAFWODdQs9YnviJWdAPNqTI8ramrqWDnAA";
final AudioPlayer player = AudioPlayer();

void main() async {
  runApp(const MyApp());
  await flutterTts.stop();
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,

      // 1. DEFINE YOUR DARK THEME
      darkTheme: ThemeData.dark().copyWith(
        // Customize the AppBar theme for dark mode
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.black, // A deep black for the AppBar
          foregroundColor: Colors.white, // White color for title and icons
          elevation: 0,
        ),
        // Customize the main background color
        scaffoldBackgroundColor: const Color(0xFF121212),
        // Customize the floating action button's color
        floatingActionButtonTheme: const FloatingActionButtonThemeData(
          backgroundColor: Colors.blue, // A nice accent color
        ),
      ),

      // 2. SET THE THEME MODE TO DARK
      // This forces the app to use the darkTheme you defined above.
      themeMode: ThemeMode.dark,

      home: const SpeechToTextExample(),
    );
  }
}

class SpeechToTextExample extends StatefulWidget {
  const SpeechToTextExample({super.key});

  @override
  _SpeechToTextExampleState createState() => _SpeechToTextExampleState();
}

class _SpeechToTextExampleState extends State<SpeechToTextExample> {
  late stt.SpeechToText _speech;
  bool _isListening = false;
  String _text = "Press the button and start speaking";

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
  }

  Future<void> _listen() async {
    await player.stop();
    if (!_isListening) {
      bool available = await _speech.initialize(
        onStatus: (val) => print("Status: $val"),
        onError: (val) => print("Error: $val"),
      );
      if (available) {
        setState(() => _isListening = true);
        _speech.listen(
          onResult: (val) async {
            if (val.finalResult) {
              setState(() {
                _text = val.recognizedWords;
              });
              print("✅ Final recognized text: ${val.recognizedWords}");

              if (val.recognizedWords != "") {
                String result = await askPesu(val.recognizedWords);
                setState(() {
                  _text = result;
                });
                print(result);
                await speakText(result);
              }
            }
          },
        );
      }
    } else {
      setState(() => _isListening = false);
      _speech.stop();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title:
        Text("AskPESU"),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(
            _text,
            style: const TextStyle(fontSize: 24.0),
            textAlign: TextAlign.center,
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _listen,
        child: Icon(_isListening ? Icons.mic : Icons.mic_none),
      ),
    );
  }
}

Future<String> askPesu(String query) async {
  print("requested");
  const String baseUrl = "http://10.7.19.117:5050/app";

  try {
    final response = await http.post(
      Uri.parse(baseUrl),
      headers: {
        "accept": "application/json",
        "Content-Type": "application/json",
      },
      body: jsonEncode({"query": query}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['answer'];
    } else {
      return "Error: ${response.statusCode} ${response.body}";
    }
  } catch (e) {
    return "Exception: $e";
  }
}

Future<http.Response> getAudioBytes(
    String text,
    String voicetype,
    String code,
    ) {
  String url =
      "https://texttospeech.googleapis.com/v1/text:synthesize?key=$_apikey";

  var body = jsonEncode({
    "input": {"text": text},
    "voice": {"languageCode": code, "name": voicetype, "ssmlGender": "FEMALE"},
    "audioConfig": {"audioEncoding": "MP3"},
  });

  var response = http.post(
    Uri.parse(url),
    headers: {"Content-type": "application/json"},
    body: body,
  );
  return response;
}

Future<void> speakText(String text) async {
  await player.stop();
  String dir = (await getApplicationDocumentsDirectory()).path;
  // Note: This voice name appears custom. Ensure it's supported by the API.
  String voice = "en-US-Chirp3-HD-Callirrhoe"; // A common, reliable voice
  String voiceCode = "en-US";
  File file = File(
    "$dir/" + "answer" + ".mp3",
  );
  try {
    var response = await getAudioBytes(text, voice, voiceCode);
    var jsonData = jsonDecode(response.body);
    // Check if the response contains the audio content before decoding
    if (jsonData['audioContent'] != null) {
      String audioBase64 = jsonData['audioContent'];
      Uint8List bytes = base64Decode(audioBase64);
      await file.writeAsBytes(bytes);
      await player.play(DeviceFileSource(file.path));
    } else {
      // Handle cases where audio content is not returned
      print("Error: Audio content not found in API response.");
      print("Response body: ${response.body}");
    }
  } catch (e) {
    print("An error occurred during text-to-speech processing: $e");
  }
}
