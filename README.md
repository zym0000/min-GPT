# min-GPT
一个简单的GPT模型，用于学习和研究

依赖
python > 3

训练前模型输出100字符：
Sr?qP-QWktXoL&jLDJgOLVz'RIoDqHdhsV&vLLxatjscMpwLERSPyao.qfzs$Ys$zF-w,;eEkzxjgCKFChs!iWW.ObzDnxA Ms$3

训练后模型输出：
And trust things to end, I' follow enter.
There little pery, to his house of his with three
Scowling and reigns; take the where he absole kon
the envidestory, here's doom and shall you
heart a world wrongfully was them.

EXTPA:
Go thou art my lords, pray, give that enough secret
so appointing to thy praydress good, I know
bound of that, and I belike a grave done
to Richard' us: dismiss mains my fresh
To perfight again, trusts upen to the resolvion,
of no glorious doing, our tragery, but I
fraunt, let thy leaving no advise services, and
fall committed thee, for my revil
Whereto the fine is words: is let justice,
a ministerity ant unto to dy death, a fine
'Gain far doth loss wially peace!'

PERDITA:
Agaim on, little prepared by my hand,
And but the king; and something to coterps:
I for I love, and let them trust wir; and,
and now in my eyes a Verian new work!
Deperdisdain--for these these name made or my primes.

PHEN ELIZABETH:
Lord High of the princes hath beet me
Thine ears him be lov

训练过程执行5000次后：
step     0 | loss: 4.3019
step   500 | loss: 1.8948
step  1000 | loss: 1.6121
step  1500 | loss: 1.4967
step  2000 | loss: 1.4159
step  2500 | loss: 1.3681
step  3000 | loss: 1.3163
step  3500 | loss: 1.2829
step  4000 | loss: 1.2666
step  4500 | loss: 1.2384


note:
现在只是单字符版本，实际情况下要比这个复杂的多。
上面结果是在GPU下进行训练的，如果在CPU下运行，可能达不到预期

使用方式
python train.py

运行笔记本
选项 1：Google Colab
每个笔记本都可以通过点击笔记本标题中的“在 Google Colab 中打开”徽章直接在 Google Colab 中打开

选项 2：本地 Jupyter 服务器
要在本地运行笔记本，请按照以下步骤操作：
1.python -m venv venv
2.source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install dependencies  安装依赖
pip install -r requirements.txt

4.启动 Jupyter Notebook
jupyter notebook

5.在 Jupyter 界面中导航到 notebooks 目录，并选择您想要运行的笔记本

