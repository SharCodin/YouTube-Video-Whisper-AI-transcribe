import gradio as gr
import whisper
from pytube import YouTube

def get_audio(url):
    yt = YouTube(url)
    return yt.streams.filter(only_audio=True)[0].download(filename="tmp.mp4")

def get_transcript(url, model_size, lang, format):

    model = whisper.load_model(model_size)

    if lang == "None":
        lang = None
    
    result = model.transcribe(get_audio(url), fp16=False, language=lang)

    output_text = None
    if format == "None":
        output_text = result["text"]
    elif format == ".srt":
        output_text = format_to_srt(result["segments"])

    final_output = ""
    word_count = 0
    for word in output_text.split():
        if word_count > 1000:
            final_output += "\n Summarize the above in bullet points."
            word_count = 0
            final_output += "\n\n"
        final_output += word
        final_output += " "
        word_count += 1
    final_output += "\n Summarize the above in bullet points."
    return final_output

def format_to_srt(segments):
    output = ""
    for i, segment in enumerate(segments):
        output += f"{i + 1}\n"
        output += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        output += f"{segment['text']}\n\n"
    return output

def format_timestamp(t):
    hh = t//3600
    mm = (t - hh*3600)//60
    ss = t - hh*3600 - mm*60
    mi = (t - int(t))*1000
    return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d},{int(mi):03d}"


langs = ["None"] + sorted(list(whisper.tokenizer.LANGUAGES.values()))
model_size = list(whisper._MODELS.keys())

with gr.Blocks() as demo:

    with gr.Row():

        with gr.Column():

            with gr.Row():
                url = gr.Textbox(placeholder='Youtube video URL', label='URL')

            with gr.Row():

                model_size = gr.Dropdown(choices=model_size, value='tiny', label="Model")
                lang = gr.Dropdown(choices=langs, value="english", label="Language (Optional)")
                format = gr.Dropdown(choices=["None", ".srt"], value="None", label="Timestamps? (Optional)")

            with gr.Row():
                gr.Markdown("Larger models are more accurate, but slower. For 1min video, it'll take ~30s (tiny), ~1min (base), ~3min (small), ~5min (medium), etc.")
                transcribe_btn = gr.Button('Transcribe')

        with gr.Column():
            outputs = gr.Textbox(placeholder='Transcription of the video', label='Transcription')

    transcribe_btn.click(get_transcript, inputs=[url, model_size, lang, format], outputs=outputs)

demo.launch(debug=True, share=True)
