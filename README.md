## LIBODM

[Manual](https://github.com/murongxixi/libodm/wiki/Manual)　　[使用说明](https://github.com/murongxixi/libodm/wiki/%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E)

LIBODM is a cross-platform package for the **o**ptimal margin **d**istribution **m**achine (ODM), which aims to achieve better binary classification performance by explicitly optimizing the margin distribution [1]. It currently contains one dual solver supporting four different kernels and two primal solvers exploiting linear kernel:

| problem |           solver           |               kernel                |
| :-----: | :------------------------: | :---------------------------------: |
|  dual   |  dual coordinate descent   | linear / polynomial / rbf / sigmoid |
| primal  | trust region Newton method |               linear                |
| primal  |            svrg            |               linear                |

<br>

All the solvers are implemented by C++, thus it can be directly called from cmd, but we also provide two friendly use interfaces, i.e., python and octave / matlab. The package has been tested on Windows / Linux / MacOS. To get started, please read the documents in the wiki pages.

<br>

If you find libodm helpful, please cite it as

[1] Teng Zhang and Zhi-Hua Zhou. [Optimal margin distribution machine](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde19odm.pdf). **IEEE Transactions on Knowledge and Data Engineering**, 32(6):1143–1156,
2019.

<br>

For any questions and comments, please feel free to send email to tengzhang@hust.edu.cn.
