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


