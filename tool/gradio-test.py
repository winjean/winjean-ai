import gradio as gr


def update(name):
    return f"Welcome to Gradio, {name}!"


with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
        with gr.Row():
            with gr.Column():
                gr.Markdown("This is a markdown component.")
                inp1 = gr.Textbox(placeholder="What is your name1?")
                inp2 = gr.Textbox(placeholder="What is your name2?")
                inp3 = gr.Textbox(placeholder="What is your name3?")
                gr.Dataframe()
        with gr.Row():
            with gr.Row():
                out1 = gr.Textbox()
                out2 = gr.Textbox()
        submit = gr.Button("submit1", variant="primary", size="lg")
    with gr.Column():
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
    with gr.Column():
        gr.Markdown("This is a markdown component.")
        submit = gr.Button("submit2", variant="primary")
    btn = gr.Button("Run")
    btn.click(fn=update, inputs=inp, outputs=out)

demo.launch(server_port=8888, share=True)

if __name__ == '__main__':
    pass
