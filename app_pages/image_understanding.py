import time
import base64
from glob import glob
import streamlit as st
from typing import Optional, Callable


def render_image_understanding_page(
    generated_images_dir: Optional[str] = None, image_generation_util: Optional[Callable] = None,
    image_understanding_util: Optional[Callable] = None
) -> None:
    """
    Renders a page in a Streamlit application for generating and displaying images using a specified AI model.
    Users can enter a prompt, select a model and image size, and specify the number of images to generate.
    The function displays the generated images and the total cost based on the selected options.
    Previously generated images in the specified directory are also displayed.

    :param generated_images_dir: Optional; the directory where generated images are stored. Defaults to None.
    :param image_generation_util: Optional; a utility function provided for image generation. This function should
                                  accept parameters for the prompt, image size, number of images to generate,
                                  directory for storing generated images, and model name. It should return a list of
                                  file paths for the generated images. Defaults to None.
    :param image_understanding_util: Optional; GPT4 Vision to understand image.
    :return: None
    """
    cols = st.columns([1, 1])
    with cols[0]:
        st.header('Image Generation with DALL·E', divider='blue')
        image_cost = {'256x256': 0.016, '512x512': 0.018, '1024x1024': 0.020}
        with st.form('image_generation'):
            prompt = st.text_area("Enter Prompt:", placeholder="Whatever your mind can imagine!")
            model_name = st.selectbox("Select model:", options=["dall-e-3", "dall-e-2"], index=0)
            image_size = st.selectbox("Image size:", options=['1024x1024', '1024x1792', '1792x1024'], index=0)
            images_to_generate = st.number_input("Images:", value=1, min_value=1, max_value=10)
            btn_cols = st.columns([2, 2, 1])
            with btn_cols[1]:
                submitted = st.form_submit_button("Generate!", type="primary")
            if submitted:
                generated_image_filepaths = image_generation_util.generate_image(
                    prompt=prompt, image_size=image_size, images_to_generate=images_to_generate,
                    generated_images_dir=generated_images_dir, model_name=model_name
                )
                st.markdown(
                    f"Total cost: <b>${round(image_cost[image_size] * images_to_generate, 2)}</b>",
                    unsafe_allow_html=True
                )
                st.image(generated_image_filepaths, caption=[prompt]*images_to_generate)

    with cols[1]:
        st.header('Image Understanding with GPT-4', divider='green')
        with st.form('image_understanding'):
            prompt = st.text_area("Enter Question:", value="What’s in this image?", height=145)
            image = st.file_uploader('Upload Image:')
            max_tokens = int(st.number_input('Max Tokens:', value=512))
            btn_cols = st.columns([2, 2, 1])
            with btn_cols[1]:
                submitted = st.form_submit_button("Explain!", type="primary")
            if submitted:
                image_bytes = base64.b64encode(image.read()).decode('utf-8')
                response = image_understanding_util.analyze_image(
                    image_bytes, prompt=prompt, max_tokens=max_tokens, streamlit_origin=True
                )
                passed = False
                try:
                    response_content = response['choices'][0]['message']['content']
                    passed = True
                except:
                    response_content = response
                st.markdown("<h6>> Model think...</h6>", unsafe_allow_html=True)
                cols = st.columns([0.2, 1, 1, 0.2])
                with cols[1]:
                    if passed:
                        message_placeholder = st.empty()
                        full_response = ""
                        for chunk in response_content.split():
                            full_response += chunk + " "
                            time.sleep(0.03)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    else:
                        st.json(response_content)
                with cols[2]:
                    st.image(image, use_column_width=True)

    st.divider()

    # display previously generated images
    with st.expander('Previously generated images'):
        all_images_filepath = glob(f"{generated_images_dir}/*.png")
        st.subheader(f'Previously generated images: {len(all_images_filepath)}', divider='red')
        st.image(all_images_filepath)

