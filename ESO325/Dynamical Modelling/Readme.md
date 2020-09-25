## This folder contains the codes related to dynamic modeling of galaxies.


Nesta etapa do projeto é realizada a modelagem dinâmica da galáxia ESO325. O modelo é obtido a partir dos dados de espectroscopia de campo integral obtidos pelo MUSE e também dos dados fotométricos na banda F814W obtidos com o HST.

Para isso, será necessário resolver as equações de Jeans que descrevem a dinâmica de galáxias. A abordagem seguida aqui é a mesma sugerida por [Michele Cappellari](https://www-astro.physics.ox.ac.uk/~mxc/) em seus artigos abaixo linkados. Essa abordagem não só se mostrou robusta ([Cappellari M., 2016, ARA&A, 54, 597](https://ui.adsabs.harvard.edu/abs/2016ARA%26A..54..597C/exportcitation)) como também possui a facilidade de possuir implementação em python ([**JAMPY**](https://pypi.org/project/jampy/)). Por isso, certifique-se de possuir os pacotes necessários (inidicados na página principal) previamente instalados.

O primeiro passo é obter um mapa cinemático da galáxia a partir dos espectros. Para isso é realizada a síntese espectral em cada spaxel do IFU. A síntese é feita utilizando o pacote **pPXF**, e a documentação e passos seguidos se encontram descritos na pasta correspondente. O resultado final obtido é um mapa com a dispersão de velocidades em cada spaxel, bem como a velocidade associada e o erro.

    PS: A manipulação dos cubos de dados é em grande parte facilitada pelo uso do pacote mpdaf.
O passo seguinte seria construir o modelo dinâmico usando tanto dados do HST como o mapa de dispersão construído anteriormente. Contudo, nas imagens provenientes do HST (F814W) há a forte presença de arcos gravitacionais dentro do halo galático e que podem afetar o modelo dinâmico. Por essa razão se faz necessário reduzirmos novamente os dados fotométricos, de modo a eliminar o máximo possível a influência dos arcos na luz da galáxia lente.

A remoção dos arcos, bem como a adequação da imagem para construção do modelo dinâmico se econtra na pasta HST DATA REDUCTION. Novamente, o passo a passo de cada etepa é descrito em um arquivo no interior da pasta e a partir de comentários ao longo dos códigos. Além disso, aproveitamos a oportunidade para explorar o contraste de cor entre as bandas F475w e F814w para detectar os arcos gravitacionais e remover a luz da galáxia lente, sendo possível desta forma obter uma imagem dos arcos sem a necessidade de assumir um perfil de luminosidade para  lente. Ao final de todos os processos desta pasta obtemos:

- imagem dos arcos gravitacionais sem a interferência de luz da lente;
- imagem interpolada da lente (F814W), sem a presença dos arcos;

Por fim, agora que obtemos um mapa de dispersão de velocidades e a imagem sem a presença dos arcos, podemos seguir para a construção do modelo dinâmico. O modelo é construído em cima da decomposição da luz da galáxia por meio do formalismo Multi-Gaussian Expansian (MGE), amplamente utilizado em estudos fotométricos e muito versátil para estudos de cinemática galática. Com base na decomposição em MGE da imagem F814W e nos dados de dispersão de velocidade obtidos a partir do MUSE, o código JAMPY é capaz de gerar em modelo dinâmico. Esse modelo pode ser auto-consistente, isto é, leva em conta apenas a massa observada a partir da luminosidade estelar; ou um modelo que agrega também uma componente de matéria escura. Neste último caso, é necessário que o perfil que descreve o halo de matéria escura seja também parametrizado seguindo o formalismo MGE. Toda essa discussão, códigos e implementação se encontra na pasta MGE and JAM, incluindo uma discussão sobre a descomposição do perfil que irá descrever a componente de matéria escura.

Ao terminar esta etapa, devemos possuir um modelo dinâmico baseado na solução das equações de Jeans e em dados observacionais provenientes do HST e MUSE. Este modelo será, mais tarde, implementado junto ao modelo da lente para gerar um único modelo auto-consistente.
