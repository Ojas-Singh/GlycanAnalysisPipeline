import streamlit as st
import pandas as pd
import numpy as np
from lib import pdb
import py3Dmol
from stmol import showmol
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as stg
import plotly.graph_objects as go
import config
import time,os
from sklearn.cluster import KMeans,MiniBatchKMeans
from scipy import stats
from sklearn import metrics
from scipy.spatial.distance import cdist
import glob
import tempfile
import zipfile
import streamlit as st
from pathlib import Path
import base64


def zip_files_in_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))

def create_zip_download(folder_path):
    temp_dir = tempfile.gettempdir()
    zip_name = "zipped_files.zip"
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
            # st.write(glycan)
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
        col1, col2 = st.columns(2)
        with col1:
            fig0 = px.scatter_3d(
                pca_df,
                x="0",
                y="1",
                z="2",
                color="i"
            )
            fig0.update_traces(marker=dict(size=2,),
                    selector=dict(mode='markers'))
            st.plotly_chart(fig0, theme="streamlit", use_conatiner_width=True)
        with col2:
            st.image(f+"/output/PCA_variance.png")
        n_dim = st.slider('PCA Dimensions to Consider :', 0, 19, 10)
        selected_columns = [str(i) for i in range(1, n_dim+1)]
        # sil_scores = [metrics.silhouette_score(pca_df[['0','1','2','3','4','5','6','7','8','9']],
        #                         # KMeans(n_clusters=k).fit(pca_df[['0','1','2','3','4','5','6','7','8','9']]).labels_
        #                         MiniBatchKMeans(compute_labels=True,n_clusters=k,init='k-means++', reassignment_ratio=0.05,max_iter=5000,batch_size=64).fit(pca_df[['0','1','2','3','4','5','6','7','8','9']]).labels_
        # ,metric='euclidean')  for k in range(2,20)]
        # st.write(sil_scores)
        n_clusters = st.slider('Clusters Number :', 0, 20, 4)
        # clustering = MiniBatchKMeans(compute_labels=True,n_clusters=n_clusters,init='k-means++', reassignment_ratio=0.05,max_iter=5000,batch_size=64).fit(pca_df[selected_columns])
        print(pca_df[selected_columns].to_numpy())
        clustering = KMeans(n_clusters).fit(pca_df[selected_columns].to_numpy(dtype=float))
        clustering_labels= clustering.labels_
        pca_df.insert(1,"cluster",clustering_labels,False)
        df.insert(1,"cluster",clustering_labels,False)
        df["cluster"] = df["cluster"].astype(str)
        pca_df["cluster"] = pca_df["cluster"].astype(str)
        
        fig1 = px.scatter(
            pca_df,
            x="0",
            y="1",
            color="cluster"
        )
        fig1.update_traces(marker=dict(size=2,),
                  selector=dict(mode='markers'))
        st.plotly_chart(fig1, theme="streamlit", use_conatiner_width=True)
        popp=[]
        pcanp = pca_df[selected_columns].to_numpy()
        for i in range(n_clusters):
            df0 = pca_df.loc[df["cluster"] ==str(i)]
            o=[]
            pp=clustering.cluster_centers_[i]
            for j in range(len(df0.iloc[:,0])):
                o.append([np.linalg.norm(np.asarray(pp[:])-pcanp[j]),df0["i"].iloc[j],df0["cluster"].iloc[j]])
                # o.append([np.linalg.norm(np.asarray(pp[:])-[pca_df[['0','1','2','3','4','5','6','7','8','9']].iloc[j]]),df0["i"].iloc[j],df0["cluster"].iloc[j]])
            o.sort()
            popp.append([o[0][1],o[0][2]])
        # st.write(popp)
        sizee=[]
        for i in range(len(popp)):
            popp[i].append(100*float(len(df.loc[(df['cluster']==str(i)),['cluster']].iloc[:]['cluster'].to_numpy())/len(df.iloc[:]['cluster'].to_numpy())))
        st.write(popp)
        clusters=[]
        zip_path = create_zip_download(f+'/clusters')
        if st.button("Create Zipped File of Clusters"):
            with open(zip_path, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:file/zip;base64,{b64}" download="{Path(zip_path).name}">Download All Clusters Zip File</a>'
                st.markdown(href, unsafe_allow_html=True)
        # for path in glob.glob(f+'/clusters/*.pdb'):
        #     clusters.append(path)
        # if len(clusters)>0:

        if st.button('Save new Cluster Center to the Database',key="process"):
            fmd=config.data_dir+glycan+"/"+glycan+".pdb"
            pdb.exportframeidPDB(fmd,popp,str(glycan))
        xax = st.selectbox(
    'Select Torsion for X axis',
    (list(df.columns.values)),key="x")
        yax = st.selectbox(
        'Select Torsion for Y axis',
        (list(df.columns.values)),key="y")
        t = np.linspace(-1, 1.2, 2000)
        psi = df[xax]
        phi = df[yax]
        x=psi
        y=phi
        fig = go.Figure(go.Histogram2dContour(
                x = x,
                y = y,
                colorscale = 'Blues'
        ))
        print()
        # fig.add_trace(go.Scatter(pca_df.loc[(pca_df['i'] in list(np.asarray(popp).T[0])).item(),['0']].iloc[:]['0'], pca_df.loc[(pca_df['i'] in list(np.asarray(popp).T[0])).item(),['1']].iloc[:]['1'],mode='markers'
        #             ))
        fig.add_trace(go.Scatter(x=psi[np.asarray(popp,dtype=int).T[0]],y=phi[np.asarray(popp,dtype=int).T[0]],mode='markers'
                    ))

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        fig2 = px.scatter(
            df,
            x=xax,
            y=yax,
            color="i",
            # color_continuous_scale="reds",
        )
        fig2.update_traces(marker=dict(size=2,),
                  selector=dict(mode='markers'))
        st.plotly_chart(fig2, theme="streamlit", use_conatiner_width=True)
        fig3 = px.scatter(
            df,
            x=xax,
            y=yax,
            color="cluster",
            # color_continuous_scale="reds",
        )
        fig3.update_traces(marker=dict(size=2,),
                  selector=dict(mode='markers'))
        st.plotly_chart(fig3, theme="streamlit", use_conatiner_width=True)
        
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