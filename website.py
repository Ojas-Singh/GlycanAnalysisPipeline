import streamlit as st
import pandas as pd
import numpy as np
from lib import pdb
from lib import clustering
import py3Dmol
from stmol import showmol
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import config
import time,os
from sklearn import metrics
import tempfile
import zipfile
from pathlib import Path
import base64
import shutil


def zip_files_in_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path+"/alpha"):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
        for root, _, files in os.walk(folder_path+"/beta"):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))

def create_zip_download(folder_path,name):
    temp_dir = tempfile.gettempdir()
    zip_name = name+".zip"
    zip_path = os.path.join(temp_dir, zip_name)
    zip_files_in_folder(folder_path, zip_path)
    return zip_path

st.set_page_config(page_title="GlycoShape", page_icon='new_logo.png', layout="wide", initial_sidebar_state="auto", menu_items=None)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
image = Image.open('logo_new.png')
st.sidebar.image(image, caption='')
glycan=""
glycan = st.sidebar.text_input(
        "Enter GLYCAM ID of the glycam...",placeholder ="LFucpa1-2DGalpb1-3DGlcpNAcb1-OH etc"
    )
# dirlist = [ item for item in os.listdir(config.data_dir) if os.path.isdir(os.path.join(config.data_dir, item)) ]
# glycan = st.sidebar.selectbox('Glycan Sequence :  ',(dirlist))
if not glycan=="":
    fold=config.data_dir+ "/"+ glycan +"/output/structure.pdb"
    f=config.data_dir+ glycan 
    pca_df = pd.read_csv(f+"/clusters/pca.csv")
    df = pd.read_csv(f+"/clusters/torsions.csv")
    with open(fold) as ifile:
        system = "".join([x for x in ifile])
        tab1, tab2, tab3 = st.tabs(["Structure", "GAP Clusters", "Manual Cluster"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.code (glycan)
                protein = pdb.parse(fold)
                xyzview = py3Dmol.view()
                xyzview.addModelsAsFrames(system)
                xyzview.setStyle({'stick':{'color':'spectrum'}})
                xyzview.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "lightgrey"},{"hetflag": False})
                xyzview.setBackgroundColor('#FFFFFF')
                xyzview.zoomTo()
                showmol(xyzview,height=800,width=900)
            with col2:
                btn = st.download_button(
                            label="Download PDB Structure",
                            data=system,
                            file_name=glycan+".pdb",
                            mime='text/csv'
                        )
        with tab2:
            fig0 = px.scatter_3d(pca_df,x="0",y="1",z="2",color="i")
            fig0.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
            st.plotly_chart(fig0, theme="streamlit", use_conatiner_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(f+"/output/PCA_variance.png")
            with col2:
                st.image(f+"/output/Silhouette_Score.png")

            zip_path = create_zip_download(f+'/clusters',glycan)
            isExist = os.path.exists(f+'/clusters')
            if isExist:
                st.info('Fetched Cluster from Database!')
                
                with open(f+'/clusters/info.txt', 'r') as file:
                    lines = file.readlines()
                    exec(lines[0])
                    exec(lines[1])
                    exec(lines[2])
                selected_columns = [str(i) for i in range(1, n_dim+1)]
                st.write('Dimensions considered for the clustering is ',n_dim)
                if st.button("Create Zipped File of Clusters"):
                    with open(zip_path, "rb") as f:
                        bytes_data = f.read()
                        b64 = base64.b64encode(bytes_data).decode()
                        href = f'<a href="data:file/zip;base64,{b64}" download="{Path(zip_path).name}">Download All Clusters Zip File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                # clustering_labels,pp = clustering.best_clustering(n_clus,pca_df[selected_columns])
                # pca_df.insert(1,"cluster",clustering_labels,False)
                # df.insert(1,"cluster",clustering_labels,False)
                df["cluster"] = df["cluster"].astype(str)
                pca_df["cluster"] = pca_df["cluster"].astype(str)
                fig1 = px.scatter_3d(pca_df,x="0",y="1",z="2",color="cluster")
                fig1.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
                st.plotly_chart(fig1, theme="streamlit", use_conatiner_width=True)
                st.write(popp)

                list_torsion = list(df.columns.values)
                list_torsion.pop(0)
                list_torsion.pop(0)
                xax = st.selectbox(
                'Select Torsion for X axis',
                (list_torsion),key="x",index=1)
                yax = st.selectbox(
                'Select Torsion for Y axis',
                (list_torsion),key="y",index=2)
                t = np.linspace(-1, 1.2, 2000)
                psi = df[xax]
                phi = df[yax]
                x=psi
                y=phi
                fig = go.Figure(go.Histogram2dContour(x = x,y = y,colorscale = 'Blues'))
                fig.add_trace(go.Scatter(x=psi[np.asarray(popp,dtype=int).T[0]],y=phi[np.asarray(popp,dtype=int).T[0]],mode='markers'))
                fig.update_yaxes(range=(-180, 180),constrain='domain')               
                fig.update_xaxes(range=(-180, 180),constrain='domain')
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                fig2 = px.scatter(df,x=xax,y=yax,color="i",)
                fig2.update_yaxes(range=(-180, 180),constrain='domain')               
                fig2.update_xaxes(range=(-180, 180),constrain='domain')
                fig2.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
                st.plotly_chart(fig2, theme="streamlit", use_conatiner_width=True)
                fig3 = px.scatter(df,x=xax,y=yax,color="cluster",)
                fig3.update_yaxes(range=(-180, 180),constrain='domain')               
                fig3.update_xaxes(range=(-180, 180),constrain='domain')
                fig3.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
                # fig3.add_trace(go.Scatter(x=psi[np.asarray(popp,dtype=int).T[0]],y=phi[np.asarray(popp,dtype=int).T[0]],mode='markers'))

                st.plotly_chart(fig3, theme="streamlit", use_conatiner_width=True)



        with tab3:
            st.write("Under Development!")
            # fig0 = px.scatter_3d(pca_df,x="0",y="1",z="2",color="i")
            # fig0.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
            # st.plotly_chart(fig0, theme="streamlit", use_conatiner_width=True)
            # col1, col2 = st.columns(2)
            # with col1:
            #     st.image(f+"/output/PCA_variance.png")
            # with col2:
            #     st.image(f+"/output/Silhouette_Score.png")

            # zip_path = create_zip_download(f+'/clusters',glycan)
            # isExist = os.path.exists(f+'/clusters')
            # if isExist:
            #     st.info('Fetched Cluster from Database!')
            #     selected_columns = [str(i) for i in range(1, config.n_dim+1)]
            #     with open(f+'/output/info.txt', 'r') as file:
            #         lines = file.readlines()
            #         exec(lines[0])
            #         exec(lines[1])
            #         exec(lines[2])
            #     st.write('Dimensions considered for the clustering is ',n_dim)
            #     if st.button("Create Zipped File of Clusters"):
            #         with open(zip_path, "rb") as f:
            #             bytes_data = f.read()
            #             b64 = base64.b64encode(bytes_data).decode()
            #             href = f'<a href="data:file/zip;base64,{b64}" download="{Path(zip_path).name}">Download All Clusters Zip File</a>'
            #             st.markdown(href, unsafe_allow_html=True)
            #     clustering_labels,pp = clustering.best_clustering(n_clus,pca_df[selected_columns])
            #     pca_df.insert(1,"cluster",clustering_labels,False)
            #     df.insert(1,"cluster",clustering_labels,False)
            #     df["cluster"] = df["cluster"].astype(str)
            #     pca_df["cluster"] = pca_df["cluster"].astype(str)
            #     fig1 = px.scatter_3d(pca_df,x="0",y="1",z="2",color="cluster")
            #     fig1.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
            #     st.plotly_chart(fig1, theme="streamlit", use_conatiner_width=True)

            #     list_torsion = list(df.columns.values)
            #     list_torsion.pop(0)
            #     list_torsion.pop(0)
            #     xax = st.selectbox(
            #     'Select Torsion for X axis',
            #     (list_torsion),key="x",index=0)
            #     yax = st.selectbox(
            #     'Select Torsion for Y axis',
            #     (list_torsion),key="y",index=1)
            #     t = np.linspace(-1, 1.2, 2000)
            #     psi = df[xax]
            #     phi = df[yax]
            #     x=psi
            #     y=phi
            #     fig = go.Figure(go.Histogram2dContour(x = x,y = y,colorscale = 'Blues'))
            #     fig.add_trace(go.Scatter(x=psi[np.asarray(popp,dtype=int).T[0]],y=phi[np.asarray(popp,dtype=int).T[0]],mode='markers'))
            #     fig.update_yaxes(range=(-180, 180),constrain='domain')               
            #     fig.update_xaxes(range=(-180, 180),constrain='domain')
            #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            #     fig2 = px.scatter(df,x=xax,y=yax,color="i",)
            #     fig2.update_yaxes(range=(-180, 180),constrain='domain')               
            #     fig2.update_xaxes(range=(-180, 180),constrain='domain')
            #     fig2.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
            #     st.plotly_chart(fig2, theme="streamlit", use_conatiner_width=True)
            #     fig3 = px.scatter(df,x=xax,y=yax,color="cluster",)
            #     fig3.update_yaxes(range=(-180, 180),constrain='domain')               
            #     fig3.update_xaxes(range=(-180, 180),constrain='domain')
            #     fig3.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
            #     st.plotly_chart(fig3, theme="streamlit", use_conatiner_width=True)

            #     st.warning("This will delete the current Clusters in Database.")
            #     if st.button('Re-Calculate Clusters',key="re"):
            #         shutil.rmtree(f+'/clusters')
            #         st.info('Current clusters is successfully deleted')
            #         st.info('Please, refresh this page to proceed!')
            # else:
            #     st.warning('Clusters does not exist in Database, Please make clusters below!', icon=None)
            #     n_dim = st.slider('PCA Dimensions to Consider :', 0, 19, 10,key="dim")
            #     selected_columns = [str(i) for i in range(1, n_dim+1)]
            #     if st.button('Calculate silhouette_score',key="score"):
            #         for i in range(2-2,12-2):
            #             st.write("Cluster Number ",i+2," : ",metrics.silhouette_score(pca_df[selected_columns],
            #                                 clustering.best_clustering(i+2,pca_df[selected_columns])[0]
            #         ,metric='euclidean'))
            #     n_clusters = st.slider('Clusters Number :', 0, 20, 4,key="pepe")
            #     clustering_labels,pp = clustering.best_clustering(n_clusters,pca_df[selected_columns])
            #     pca_df.insert(1,"cluster",clustering_labels,False)
            #     df.insert(1,"cluster",clustering_labels,False)
            #     df["cluster"] = df["cluster"].astype(str)
            #     pca_df["cluster"] = pca_df["cluster"].astype(str)
            #     if st.button('Calculate KDE Centers of Clusters',key="kdee"):
            #         popp = clustering.kde_c(n_clusters,pca_df,selected_columns)        
            #         st.write(popp)
            #         fmd=config.data_dir+glycan+"/"+glycan+".pdb"
            #         output_cluster_folder = config.data_dir + glycan + "/clusters/"
            #         pdb.exportframeidPDB(fmd,popp,output_cluster_folder)
            #         clusters=[]

            #         with open(f+'/clusters/info.txt', 'w') as file:
            #             file.write(f"n_clusters = {n_clusters}\n")
            #             file.write(f"popp = {list(popp)}\n")
            #             file.write(f"n_dim = {n_dim}\n")

                    
            #         st.info('Please refresh this Page to View the Result!')


else:
    st.title("Welcome to GLYCOSHAPE Database!")
    dirlist = [ item for item in os.listdir(config.data_dir) if os.path.isdir(os.path.join(config.data_dir, item)) ]
    col1, col2= st.columns(2)
    col1.metric("Total Glycans",len(dirlist))
    col2.metric("Simulation Time", str(len(dirlist)*1.5) +"  μs")
    
    st.write("Please Enter Glycam ID of the Glycan in the left sidebar from below list!")
    for i in dirlist:
        st.code (i)
