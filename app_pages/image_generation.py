import streamlit as st


def render_image_generation_page(generated_images_dir=None, image_generation_util=None):
    """

    """
    st.header('Image Generation with DALL·E', divider='blue')
    with st.form('image_generation'):
        cols = st.columns([1, 0.4, 0.2])
        with cols[0]:
            prompt = st.text_input("Enter prompt:")
        with cols[1]:
            image_size = st.selectbox("Image size:", options=['256x256', '512x512', '1024x1024'], index=1)
        with cols[2]:
            images_to_generate = st.number_input("Images:", value=1, min_value=1, max_value=10)

        submitted = st.form_submit_button("Generate!", type="primary")
        if submitted:
            generated_image_filepaths = image_generation_util.generate_image(
                prompt=prompt, image_size=image_size, images_to_generate=images_to_generate,
                generated_images_dir=generated_images_dir
            )
            st.image(generated_image_filepaths)
