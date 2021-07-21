# Nada dentro dessa pasta deve ser considerdo como resultado. São apenas testes de implementação. Toda precaução com o que ver!!!

## Nota ao leitor

Como forma de testar os códigos e averiguar a implementação dos mesmos, nosso primeiro objetivo é reproduzir os resultados encontrados por __[Collett et. al 2018](https://science.sciencemag.org/content/360/6395/1342)__ para a galáxia ESO325-G004 (ESO325).

Nesta pasta encontram-se os resultados do modelamento dinâmico, modelamento da lente e do modelo combinado utilizando o Emcee (implementação python para o método Markov-Chain Monte Carlo).

A descrição de cada modelo pode ser encontrada em em sua pasta, bem como comentários sobre cada etapa do modelo. Não há uma ordem "correta" para a execução (com a excessão de que o modelo combinado deve ser o último processo a ser realizado). Contudo, recomenda-se que o modelo dinâmico seja realizado primeiro (pPXF $\rightarrow$ MGE $\rightarrow$ JAM), pois durante esse processo já é possível obter a imagem dos arcos gravitacionais que serão modelados na parte posterior (Pyautolens).


## Data
Aqui daremos uma breve descrição dos dados utilizados e onde pode-se obtê-los de maneira pública.


### Dados fotométricos
Os dados fotométricos para a galáxia ESO325 foram obtidos pelo telescópio Espacial Hubble (HST) e estão disponíveis no repositório de dados públicos ligado a [Space Telescope Science Institute (STScI)](https://www.stsci.edu/), o __[Hubble Legacy Archive](https://hla.stsci.edu/hlaview.html#Inventory|filterText%3D%24filterTypes%3D|query_string=ESO325-G004&posfilename=&poslocalname=&posfilecount=&listdelimiter=whitespace&listformat=degrees&RA=205.888330&Dec=-38.176000&Radius=0.010833&inst-control=all&inst=ACS&inst=ACSGrism&inst=WFC3&inst=WFPC2&inst=NICMOS&inst=NICGRISM&inst=COS&inst=WFPC2-PC&inst=STIS&inst=FOS&inst=GHRS&imagetype=best&prop_id=&spectral_elt=&proprietary=both&preview=1&output_size=256&cutout_size=12.8|ra=&dec=&sr=&level=&image=&inst=ACS%2CACSGrism%2CWFC3%2CWFPC2%2CNICMOS%2CNICGRISM%2CCOS%2CWFPC2-PC%2CSTIS%2CFOS%2CGHRS&ds=)__.

Os dados foram obtidos durante observações realizadas nos anos de 2005 e 2007 e contam com imagens em três diferentes bandas: F814W, F475W e F625W. Abaixo sumarizamos algumas informações:

|Banda|Ano|Exp. Time (s)| Central wavelength (A)|
|:--:|:--:|:--:|:--:|
|F814W|2007|18882|8128.69|
|F475W|2007|3501|4801.57|
|F625W|2007|4650|6352.58|

Para nossos propósitos iniciais apenas as imagens nas bandas F814W e F475W serão utilizadas.


### Dados de espectroscopia
Para uma modelagem dinâmica mais robusta são necessários dados de espectroscopia de alta qualidade, e em particular, dados de espectroscopia de campo integral (IFU) são muito bem vindos, dado o seu poder de resolução espacial.

Por sorte, dados desse tipo estão disponíveis publicamente para ESO325. Os dados disponíveis foram obtidos com o Multi Unit-Spectroscopic Explorer (MUSE), um espectroscópio de campo integral de alta resolução situado no  European Southern Observatory (ESO).

Os dados públicos podem ser acessados a partir do [ESO Archive Science Portal](http://archive.eso.org/scienceportal/home). Aqui estamos interessados no cubo de dados gerado pela espectroscopia de campo integral, isto é, um cubo de dados que possui duas dimensões representando a fotometria e uma o comprimento de onda.

Os dados utilizados e presentes neste documento são resumidos abaixo.

|__ESO325 - MUSE Data Cube__|
|--|

|Spec. Range (nm)|Spec. Res.| FoV.| Sky. Res.|Exp. Time (s)|
|:--:|:--:|:--:|:--:|:--:|
|475-935.1|3027|2.37'|0.512"|2310|


