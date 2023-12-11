# GlycanAnalysisPipeline


# Installation
```
conda create -n GAP python=3.10
conda activate GAP
pip install -r requirements.txt
streamlit run cluster_streamlit.py
```

python3 -m pip -r requirements.txt
export PATH="$HOME/.local/bin:$PATH"


modify config.py 

```
python main.py
```


# Oracle firewall fix
```
sudo iptables -P INPUT ACCEPT
sudo iptables -P OUTPUT ACCEPT
sudo iptables -P FORWARD ACCEPT
sudo iptables -F
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 8080
sudo ufw allow 443
sudo ufw enable
```


#localhost/loopback
```
sudo iptables -t nat -I OUTPUT -p tcp -d 127.0.0.1 --dport 80 -j REDIRECT --to-ports 3000
```
#external
```
sudo iptables -t nat -I PREROUTING -p tcp --dport 80 -j REDIRECT --to-ports 3000
```



screen -S name
screen -r name
ctr a + d   -> to detach
pkill screen

# Citation

All of the data provided is freely available for academic use under Creative Commons Attribution 4.0 (CC BY-NC-ND 4.0 Deed) licence terms. Please contact us at elisa.fadda@mu.ie for Commercial licence. If you use this resource, please cite the following papers:

Callum M Ives and Ojas Singh et al. Restoring Protein Glycosylation with GlycoShape bioRxiv (2023).
