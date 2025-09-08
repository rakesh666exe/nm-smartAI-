import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def get_civic_info(query):
    prompt = f"Provide a comprehensive overview of the following civic topic:\n\nCivic Topic: {query}\n\n"
    return generate_response(prompt, max_length=1200)

def analyze_sentiment_and_update_dashboard(feedback_text, pos, neu, neg, total):
    prompt = f"Classify this feedback as Positive, Neutral, or Negative:\n\n{feedback_text}"
    sentiment = generate_response(prompt, max_length=50).lower()

    if "positive" in sentiment:
        pos += 1
        sentiment = "✅ Positive"
    elif "negative" in sentiment:
        neg += 1
        sentiment = "❌ Negative"
    else:
        neu += 1
        sentiment = "⚪ Neutral"

    total += 1
    return sentiment, pos, neu, neg, total, plot_chart(pos, neu, neg)

def plot_chart(pos, neu, neg):
    labels = ["Positive", "Neutral", "Negative"]
    values = [pos, neu, neg]

    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=labels, autopct="%1.1f%%", colors=["green", "gray", "red"])
    plt.title("Citizen Sentiment Distribution")
    plt.tight_layout()
    return plt.gcf()
# --- Gradio App with Landing Page ---
with gr.Blocks(theme="soft") as app:
    # --- Landing Page ---
    with gr.Group(visible=True) as landing_page:
        gr.Markdown("## 🌐 Welcome to Citizen AI Assistant")
        gr.HTML(
            """
            <div style="text-align:center; font-size:16px;">
                 <p><b>CitizenAI</b> is an intelligent assistant that empowers citizens to engage with government services, 
                provide meaningful feedback, and communicate more effectively with civic institutions.</p>
                
                <div class="checklist">
                    <p>✅ Access clear and reliable civic information</p>
                    <p>✅ Share your feedback and see sentiment analysis</p>
                    <p>✅ Visualize public opinion in real time</p>
                </div>
                <p><b>Disclaimer:</b> This tool is for informational purposes only. Always verify with official sources.</p>
            </div>
            """
        )
        enter_btn = gr.Button("🚀 Enter App", elem_id="enter-btn")

    # --- Main App (Initially Hidden) ---
    with gr.Group(visible=False) as main_app:
        gr.Markdown("## 🏛️ Citizen AI Assistant")
        gr.Markdown("🔔 *Disclaimer: Informational only. Verify with official sources.*")

        with gr.Tabs():
            with gr.TabItem("📖 Civic Information"):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Enter a civic topic",
                        placeholder="e.g., how to register to vote, recycling laws...",
                        lines=3
                    )
                    info_btn = gr.Button("🔍 Get Info")
                info_output = gr.Chatbot(label="Civic Info")

                info_btn.click(
                    lambda q: [(q, get_civic_info(q))],
                    inputs=query_input,
                    outputs=info_output
                )

            with gr.TabItem("💬 Sentiment Analysis"):
                feedback_input = gr.Textbox(
                    label="Citizen Feedback",
                    placeholder="e.g., The new bus route is fantastic!",
                    lines=4
                )
                analyze_btn = gr.Button("📊 Analyze Sentiment")
                sentiment_output = gr.Textbox(label="Result", interactive=False)

            with gr.TabItem("📈 Dashboard"):
                with gr.Row():
                    pos_box = gr.Number(label="Positive", value=0, interactive=False)
                    neu_box = gr.Number(label="Neutral", value=0, interactive=False)
                    neg_box = gr.Number(label="Negative", value=0, interactive=False)
                    total_box = gr.Number(label="Total", value=0, interactive=False)
                sentiment_chart = gr.Plot(label="Sentiment Chart")

        analyze_btn.click(
            analyze_sentiment_and_update_dashboard,
            inputs=[feedback_input, pos_box, neu_box, neg_box, total_box],
            outputs=[sentiment_output, pos_box, neu_box, neg_box, total_box, sentiment_chart]
        )

    # --- Switch from Landing → Main App ---
    enter_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[landing_page, main_app]
    )

app.launch(share=True)
