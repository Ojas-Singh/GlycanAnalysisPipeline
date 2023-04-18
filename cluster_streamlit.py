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
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))

def create_zip_download(folder_path,name):
    temp_dir = tempfile.gettempdir()
    zip_name = name+".zip"
    zip_path = os.path.join(temp_dir, zip_name)
    zip_files_in_folder(folder_path, zip_path)
    return zip_path

st.set_page_config(page_title="GlycoShape", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
image = Image.open('logo.png')
st.sidebar.image(image, caption='')
glycan=""
dirlist = [ item for item in os.listdir(config.data_dir) if os.path.isdir(os.path.join(config.data_dir, item)) ]
glycan = st.sidebar.selectbox('Glycan Sequence :  ',(dirlist))
fold=config.data_dir+ "/"+ glycan +"/output/structure.pdb"
f=config.data_dir+ glycan 
pca_df = pd.read_csv(f+"/output/pca.csv")
df = pd.read_csv(f+"/output/torsions.csv")
with open(fold) as ifile:
    system = "".join([x for x in ifile])
    tab1, tab2, tab4 = st.tabs(["Structure", "PCA Clusters", "Sampler"])
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
            selected_columns = [str(i) for i in range(1, config.n_dim+1)]
            with open(f+'/clusters/info.txt', 'r') as file:
                lines = file.readlines()
                exec(lines[0])
                exec(lines[1])
                exec(lines[2])
            st.write('Dimensions considered for the clustering is ',n_dim)
            if st.button("Create Zipped File of Clusters"):
                with open(zip_path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f'<a href="data:file/zip;base64,{b64}" download="{Path(zip_path).name}">Download All Clusters Zip File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            clustering_labels,pp = clustering.best_clustering(n_clusters,pca_df[selected_columns])
            pca_df.insert(1,"cluster",clustering_labels,False)
            df.insert(1,"cluster",clustering_labels,False)
            df["cluster"] = df["cluster"].astype(str)
            pca_df["cluster"] = pca_df["cluster"].astype(str)
            fig1 = px.scatter_3d(pca_df,x="0",y="1",z="2",color="cluster")
            fig1.update_traces(marker=dict(size=2,),selector=dict(mode='markers'))
            st.plotly_chart(fig1, theme="streamlit", use_conatiner_width=True)

            list_torsion = list(df.columns.values)
            list_torsion.pop(0)
            list_torsion.pop(0)
            xax = st.selectbox(
            'Select Torsion for X axis',
            (list_torsion),key="x",index=0)
            yax = st.selectbox(
            'Select Torsion for Y axis',
            (list_torsion),key="y",index=1)
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
            st.plotly_chart(fig3, theme="streamlit", use_conatiner_width=True)

            st.warning("This will delete the current Clusters in Database.")
            if st.button('Re-Calculate Clusters',key="re"):
                shutil.rmtree(f+'/clusters')
                st.info('Current clusters is successfully deleted')
                st.info('Please, refresh this page to proceed!')
        else:
            st.warning('Clusters does not exist in Database, Please make clusters below!', icon=None)
            n_dim = st.slider('PCA Dimensions to Consider :', 0, 19, 10,key="dim")
            selected_columns = [str(i) for i in range(1, n_dim+1)]
            if st.button('Calculate silhouette_score',key="score"):
                for i in range(2-2,12-2):
                    st.write("Cluster Number ",i+2," : ",metrics.silhouette_score(pca_df[selected_columns],
                                        clustering.best_clustering(i+2,pca_df[selected_columns])[0]
                ,metric='euclidean'))
            n_clusters = st.slider('Clusters Number :', 0, 20, 4,key="pepe")
            clustering_labels,pp = clustering.best_clustering(n_clusters,pca_df[selected_columns])
            pca_df.insert(1,"cluster",clustering_labels,False)
            df.insert(1,"cluster",clustering_labels,False)
            df["cluster"] = df["cluster"].astype(str)
            pca_df["cluster"] = pca_df["cluster"].astype(str)
            if st.button('Calculate KDE Centers of Clusters',key="kdee"):
                popp = clustering.kde_c(n_clusters,pca_df,selected_columns)        
                st.write(popp)
                fmd=config.data_dir+glycan+"/"+glycan+".pdb"
                output_cluster_folder = config.data_dir + glycan + "/clusters/"
                pdb.exportframeidPDB(fmd,popp,output_cluster_folder)
                clusters=[]

                with open(f+'/clusters/info.txt', 'w') as file:
                    file.write(f"n_clusters = {n_clusters}\n")
                    file.write(f"popp = {list(popp)}\n")
                    file.write(f"n_dim = {n_dim}\n")

                
                st.info('Please refresh this Page to View the Result!')

        

    # with tab3:
    #     fig0 = px.scatter(
    #         tsne_df,
    #         x="0",
    #         y="1",
    #         color="i",
    #         # color_continuous_scale="reds",
    #     )
    #     fig0.update_traces(marker=dict(size=2,),
    #               selector=dict(mode='markers'))
    #     st.plotly_chart(fig0, theme="streamlit", use_conatiner_width=True)
    #     n_clusters = st.slider('How many clusters?', 0, 50, 10,key="tsne")
    #     clustering = KMeans(n_clusters).fit(tsne_df[['0','1','2']])
    #     clustering_labels= clustering.labels_
    #     tsne_df.insert(1,"cluster2",clustering_labels,False)
    #     df.insert(1,"cluster2",clustering_labels,False)
        
    #     df["cluster2"] = df["cluster2"].astype(str)
    #     tsne_df["cluster2"] = tsne_df["cluster2"].astype(str)
        
    #     fig1 = px.scatter_3d(
    #         tsne_df,
    #         x="0",
    #         y="1",
    #         z="2",
    #         color="cluster2",
    #         # color_continuous_scale="reds",
    #     )
    #     fig1.update_traces(marker=dict(size=2,),
    #               selector=dict(mode='markers'))
    #     st.plotly_chart(fig1, theme="streamlit", use_conatiner_width=True)
    #     popp=[]
    #     for i in range(n_clusters):
    #         df0 = tsne_df.loc[df["cluster2"] ==str(i)]
    #         o=[]
    #         pp=clustering.cluster_centers_[i]
    #         for j in range(len(df0.iloc[:,0])):
    #             o.append([np.linalg.norm(np.asarray(pp[:2])-[df0["0"].iloc[j],df0["1"].iloc[j]]),df0["i"].iloc[j]])
    #         o.sort()
    #         popp.append(o[0][1])
    #     st.write(popp)

    #     xax = st.selectbox(
    # 'Select Torsion for X axis',
    # (list(df.columns.values)),key="x")
    #     yax = st.selectbox(
    #     'Select Torsion for Y axis',
    #     (list(df.columns.values)),key="y")
    #     t = np.linspace(-1, 1.2, 2000)
    #     psi = df[xax]
    #     phi = df[yax]
    #     x=psi
    #     y=phi
    #     fig = go.Figure(go.Histogram2dContour(
    #             x = x,
    #             y = y,
    #             colorscale = 'Blues'
    #     ))
    #     fig.add_trace(go.Scatter(x=x[popp], y=y[popp],mode='markers'
    #                 ))

    #     st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    #     fig2 = px.scatter(
    #         df,
    #         x=xax,
    #         y=yax,
    #         color="i",
    #         # color_continuous_scale="reds",
    #     )
    #     fig2.update_traces(marker=dict(size=2,),
    #               selector=dict(mode='markers'))
    #     st.plotly_chart(fig2, theme="streamlit", use_conatiner_width=True)
    #     from scipy import stats
        
    #     fig3 = px.scatter(
    #         df,
    #         x=xax,
    #         y=yax,
    #         color="cluster2",
    #         # color_continuous_scale="reds",
    #     )
    #     fig3.update_traces(marker=dict(size=2,),
    #               selector=dict(mode='markers'))
    #     st.plotly_chart(fig3, theme="streamlit", use_conatiner_width=True)
        

    # with tab4:
    #     # clustering = KMeans(50).fit(pca_df[['X','Y']])
    #     # clustering_labels= clustering.labels_
    #     # pca_df.insert(1,"cluster",clustering_labels,False)
    #     # df.insert(1,"cluster",clustering_labels,False)
        
    #     fig1 = px.scatter(
    #         pca_df,
    #         x="0",
    #         y="1",
    #         color="cluster",
    #         # color_continuous_scale="reds",
    #     )
    #     fig1.update_traces(marker=dict(size=2,),
    #               selector=dict(mode='markers'))
    #     st.plotly_chart(fig1, theme="streamlit", use_conatiner_width=True)
    #     cluster1= st.selectbox(
    # 'From cluster?',
    # (list(range(50))),key="clu")
    #     torsionrange = st.slider('Random torsion range?', 0, 10, 3)
    #     import os
    #     try:
    #         os.remove('output/wig.pdb') 
    #     except:
    #         pass
    #     if st.button('Process',key="process"):
    #         G = pdb.parse("output/cluster/"+str(cluster1)+".pdb")
    #         G= pdb.to_DF(G)
    #         loaded = np.load('data/bisecting/bisecting_torparts.npz',allow_pickle=True)
    #         Garr = G[['X','Y','Z']].to_numpy(dtype=float)
    #         # tormeta = loaded["b"]

    #         # torsions = loaded["c"]

    #         torsionpoints = loaded["a"]
    #         torsionparts  = loaded["b"]
    #         # torsionparts = np.asarray(torsionparts)
    #         # torsionpoints= np.asarray(torsionpoints)
    #         molecules = []
    #         for idx in range(20):
    #             Garr1 = algo.Garrfromtorsiondemo(Garr,torsionpoints,torsionrange,torsionparts)
    #             Gn =  pd.DataFrame(Garr1, columns = ['X','Y','Z'])
    #             G.update(Gn)
    #             g1 = pdb.exportPDBmulti('output/wig.pdb',pdb.to_normal(G),idx)
    #         with open('output/wig.pdb') as ifile:
    #             systemx = "".join([x for x in ifile])
    #             xyzview1 = py3Dmol.view()
    #             xyzview1.addModelsAsFrames(systemx)
    #             xyzview1.setStyle({'stick':{'color':'spectrum'}})
    #             # xyzview1.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "lightgrey"},{"hetflag": False})

    #             xyzview1.setBackgroundColor('#FFFFFF') 
    #             xyzview1.zoomTo()
    #             xyzview1.animate({'loop': "forward"})
    #             xyzview1.show()
    #             showmol(xyzview1,height=800,width=900)