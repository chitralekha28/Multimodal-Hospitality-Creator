import os
import torch
import chromadb
import gradio as gr
from google import genai
from diffusers import StableDiffusionPipeline
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY as environment variable.")

client = genai.Client(api_key=api_key)

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")



documents = [
    """
    Eco resorts integrate local materials like volcanic stone,
    passive cooling strategies, and open-air pavilions to enhance
    natural ventilation and climate harmony.
    """,
    """
    Sustainable hospitality architecture includes rainwater harvesting,
    solar integration, geothermal systems, and visible sustainability
    elements as part of aesthetic identity.
    """,
    """
    Cliffside resorts use terraced structural planning,
    cascading platforms, and infinity pools aligned with natural horizons.
    """,
    """
    Biophilic design blends indoor and outdoor spaces through
    green walls, courtyards, natural light, and organic materials.
    """
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client(Settings())
collection = chroma_client.get_or_create_collection("hospitality_docs")

if collection.count() == 0:
    for i, doc in enumerate(documents):
        embedding = embedding_model.encode(doc).tolist()
        collection.add(
            documents=[doc],
            embeddings=[embedding],
            ids=[str(i)]
        )


def retrieve_context(query):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return "\n".join(results["documents"][0])


def enhance_prompt(user_prompt, context):
    prompt = f"""
    You are a hospitality architecture expert.

    Context:
    {context}

    Enhance this concept into a cinematic architectural visualization prompt:

    {user_prompt}
    """

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def generate_narrative(enhanced_prompt):
    prompt = f"""
    Write a professional hospitality project description:

    {enhanced_prompt}
    """

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text



def generate_images(enhanced_prompt):

    base_prompt = f"""
    ultra realistic architectural photography,
    luxury resort building clearly visible,
    {enhanced_prompt},
    professional architectural render,
    sharp focus, 8k
    """

    negative_prompt = """
    empty landscape, no building,
    distorted, surreal, blurry,
    deformed structure
    """

    views = [
        "aerial drone shot",
        "eye level facade view",
        "wide landscape perspective",
        "poolside perspective"
    ]

    images = []

    for view in views:
        full_prompt = base_prompt + ", " + view

        image = pipe(
            full_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=9,
            num_inference_steps=15
        ).images[0]

        images.append(image)

    return images


def hospitality_creator(user_input):

    yield "⏳ Retrieving architectural knowledge...", "", []

    context = retrieve_context(user_input)

    yield "🧠 Enhancing cinematic prompt...", "", []

    enhanced = enhance_prompt(user_input, context)

    yield enhanced, "✍️ Generating professional narrative...", []

    narrative = generate_narrative(enhanced)

    yield "🎨 Rendering architectural visuals...", enhanced, narrative, []

    images = generate_images(enhanced)

    yield enhanced, narrative, images



with gr.Blocks() as app:

    gr.Markdown("# 🏨 Multimodal Hospitality Creator")
    gr.Markdown("AI-powered hospitality concept visualization using RAG + Gemini + Stable Diffusion")

    user_input = gr.Textbox(
        label="Enter Your Hospitality Concept",
        lines=3,
        placeholder="A luxury desert eco resort with sand dunes..."
    )

    generate_btn = gr.Button("Generate Concept")
    status_output = gr.Markdown()


    with gr.Tabs():
        with gr.TabItem("Enhanced Cinematic Prompt"):
            enhanced_output = gr.Markdown()

        with gr.TabItem("Project Narrative"):
            narrative_output = gr.Markdown()

        with gr.TabItem("Generated Views"):
            image_output = gr.Gallery(columns=2)

    generate_btn.click(
    fn=hospitality_creator,
    inputs=user_input,
    outputs=[
    status_output,
    enhanced_output,
    narrative_output,
    image_output
]
,
    show_progress=True
)

# Run App
app.launch()