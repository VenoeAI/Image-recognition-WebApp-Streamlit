import streamlit as st
import pandas as pd
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #e5e5f7;
opacity: 0.8;
background-image:  linear-gradient(30deg, #8bbf6a 12%, transparent 12.5%, transparent 87%, #8bbf6a 87.5%, #8bbf6a), linear-gradient(150deg, #8bbf6a 12%, transparent 12.5%, transparent 87%, #8bbf6a 87.5%, #8bbf6a), linear-gradient(30deg, #8bbf6a 12%, transparent 12.5%, transparent 87%, #8bbf6a 87.5%, #8bbf6a), linear-gradient(150deg, #8bbf6a 12%, transparent 12.5%, transparent 87%, #8bbf6a 87.5%, #8bbf6a), linear-gradient(60deg, #8bbf6a77 25%, transparent 25.5%, transparent 75%, #8bbf6a77 75%, #8bbf6a77), linear-gradient(60deg, #8bbf6a77 25%, transparent 25.5%, transparent 75%, #8bbf6a77 75%, #8bbf6a77);
background-size: 20px 35px;
background-position: 0 0, 0 0, 10px 18px, 10px 18px, 0 0, 10px 18px;
}
</style>
"""
#st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Industry Standard :blue[Image Classification AI].')
st.subheader('This program makes use of the ClarifAI API developed by :red[Matthew Zeiler] to help classify objects in image data and other features, those features can be accesed through the AI use cases.')

st.sidebar.title('Use Case')
use_case = st.sidebar.radio('Available Use Cases', ('Image Recognition','Image Moderation'))

st.info('Select AI use case from sidebar')

if use_case == 'Image Recognition':
    st.sidebar.image('imgrec.png')
    st.sidebar.info('Identifies a variety of concepts in images and video including objects, themes, and more. Trained with over 10,000 concepts and 20M images.')
    st.subheader('Image recognition model in use...')
    api_key = st.text_input('#ClarifAI API key', '')

    image = st.file_uploader("Upload Image",['JPG','PNG'])
    if image != None:
        st.image(image)

    YOUR_CLARIFAI_API_KEY = api_key

    button = st.button('Confirm')

    if button == True:

        if YOUR_CLARIFAI_API_KEY != '' and image != None:

            # This is how you authenticate.
            metadata = (("authorization", f"Key {YOUR_CLARIFAI_API_KEY}"),)

            request = service_pb2.PostModelOutputsRequest(
                # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
                model_id="general-image-recognition",
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(image=resources_pb2.Image(base64=image.getvalue()))
                    )
                ],
            )
            response = stub.PostModelOutputs(request, metadata=metadata)

            if response.status.code != status_code_pb2.SUCCESS:
                print(response)
                raise Exception(f"Request failed, status code: {response.status}")

            names = []
            confidence = []
            
            for concept in response.outputs[0].data.concepts:
                names.append(concept.name)
                confidence.append(concept.value)

            df = pd.DataFrame({
                'Concept Name': names,
                'Model Confidence': confidence
            })

            st.table(df)

if use_case == 'Image Moderation':
    st.sidebar.image('moder.jpg')
    st.sidebar.info('Recognizes inappropriate content in images and video containing concepts: gore, drug, explicit, suggestive, and safe.')
    st.subheader('Image moderation model in use...')
    api_key = st.text_input('#ClarifAI API key', '')

    image = st.file_uploader("Upload Image",['JPG','PNG'])
    if image != None:
        st.image(image)

    YOUR_CLARIFAI_API_KEY = api_key

    button = st.button('Confirm')

    if button == True:

        if YOUR_CLARIFAI_API_KEY != '' and image != None:

            # This is how you authenticate.
            metadata = (("authorization", f"Key {YOUR_CLARIFAI_API_KEY}"),)

            request = service_pb2.PostModelOutputsRequest(
                # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
                model_id="moderation-recognition",
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(image=resources_pb2.Image(base64=image.getvalue()))
                    )
                ],
            )
            response = stub.PostModelOutputs(request, metadata=metadata)

            if response.status.code != status_code_pb2.SUCCESS:
                print(response)
                raise Exception(f"Request failed, status code: {response.status}")

            names = []
            confidence = []
            
            for concept in response.outputs[0].data.concepts:
                names.append(concept.name)
                confidence.append(concept.value)

            df = pd.DataFrame({
                'Concept Name': names,
                'Model Confidence': confidence
            })

            st.table(df)

 
