# Speech Enhancement by Self-supervised Audio Transformer
The repository is co-authored by Chi-Chang Lee and Shu-wen Yang

### Set workspace
```
export WORKSPACE=/home/leo
cd $WORKSPACE
```

### Clone prejects
```
git clone https://github.com/leo19941227/Speech-Enhancement-by-S3PRL.git
git clone https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning.git
```

### Install packages
Follow the requirements of [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)

### Add python path
This step lets us be able to import `S3PRL` project as a package
- open .bashrc
```
vim $HOME/.bashrc
```
- add the following line to .bashrc
```
PYTHONPATH="$WORKSPACE/Self-Supervised-Speech-Pretraining-and-Representation-Learning:$PYTHONPATH"
```
