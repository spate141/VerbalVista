import streamlit as st


def render_image_generation_page(generated_images_dir=None, image_generation_util=None):
    """

    """
    st.header('Image Generation with DALL·E', divider='blue')
    image_cost = {'256x256': 0.016, '512x512': 0.018, '1024x1024': 0.020}
    with st.form('image_generation'):
        cols = st.columns([1, 0.4, 0.2, 0.2])
        with cols[0]:
            prompt = st.text_area("Enter prompt:")
        with cols[1]:
            model_name = st.selectbox("Select model", options=["dall-e-3", "dall-e-2"], index=0)
        with cols[2]:
            image_size = st.selectbox("Image size:", options=['1024x1024', '1024x1792', '1792x1024'], index=0)
        with cols[3]:
            images_to_generate = st.number_input("Images:", value=1, min_value=1, max_value=10)

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
