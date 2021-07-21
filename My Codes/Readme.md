# Códigos desenvolvidos e/ou adaptados para o Mestrado

Nesta pasta encontram-se os códigos que foram desenvolvidos ou adaptados para o Mestrado. Alguns deles são inclusões feitas em códigos já existentes, e outros são códigos novos desenvolvidos para permitir o modelamento conjunto de lentes gravitacionais e dinâmica de galáxias. Há ainda os códigos _CombinedModel.py_ e _dyCombinedModel.py_, que são os códigos externos que permitem a utilização de um sampler Bayesiano (emcee e dynesty, respectivamente) para explorarmos o espaço de parâmetros. O arquivo _My_Jampy.py_ é apenas um wrapper do código original Jampy, mas com uma “interface” mais simples, que permite facilmente a inclusão de matéria escura e até mesmo a realização de um sampling Bayesiano focado apenas no modelo dinâmico.

Via de regra, esses códigos são testados nas pastas Autolens tests e Jampy_tests.

Além disso, como houve a necessidade de implementar um novo modelo de massa dentro do Autolens, temos uma versão própria do arquivo _total_mass_profiles.py_ (originamente do Autolens), mas com a nossa implementação do modelo MGE. Nenhum outro perfil foi alterado, apenas o perfil descrito pelas Gaussianas foi adicionado. Além disso, você vai perceber vários arquivos com variações deste nome. Eles basicamente representam versões antigas de nossa implementação, em que buscamos melhorar o desempenho do algoritmo. A versão ideal a ser utilizada é a **total_mass_profiles.py**. A maneira de como adicionar nossa versão ao seu Autolens é descrita na sequência.



### Lens Modeling with Pyautolens develop by James



Para o modelamento da lente faremos uso do código Python Pyautolens. Por isso, certifique-se de que o pacote (com a versão recomendada já esteja instala em seu computador).

Entretanto, para um modelo  que seja auto-consistente, isto é, dinâmico+lente, faz-se necessário que o mesmo perfil de matéria (seja ele bariônico ou não) seja utilizado em ambos os modelos. Ou seja, como descrito na pasta do modelo dinâmico, para descrever o potencial (ou neste caso massa total) presente na galáxia, utilizamos a parametrização Multi-Gaussian Expansion (MGE). Portanto, é necessário que o perfil utilizado pelo Pyautolens para realizar o modelamento (inversão e reconstrução) seja também um perfil parametrizado em MGE.

Infelizmente a atual versão do Pyautolens não conta com esse tipo de parametrização, o que  nos motivou a implementar nossa própria versão dentro do código original (e open source) do Pyautolens. Uma discussão maior e com detalhes é feita no paper (em preparação e espero que logo publicado) em que descrevo o método, objetivos e resultados (também está dissertação apresentada). Contudo, um breve resumo é bem vindo:

- A mesma parametrização utilizada no modelo dinâmico é utilizada aqui, contudo é necessário que as gaussianas que descrevem a luz sejam antes convertidas em gaussianas que descrevam a densidade superficial de massa. Para isso, informamos um mass-to-light (ML) ratio que realiza essa conversão.
- Como o modelo de lente não vê como essa massa está distribuída, isto é, apenas a massa total é importante, nós convertemos essa densidade superficial de massa em uma massa total por gaussiana.
- Finalmente essa massa total por Gaussiana (juntamente com suas dispersões e axial ratio) são passadas ao modelo de lente, onde implementamos os ângulos de deflexão gerados por esse tipo de parametrização.
- Após calculados esses ângulos de deflexão e armazenados numa grid, essa grid é passada ao Pyautolens, que realiza a inversão linear dos pontos no plano da lente até o plano da fonte. A partir disso tanto a fonte quanto a lente reconstruídas podem ser obtidas.

Como adiantado acima, o Pyautolens não possuí essa parametrização em sua documentação padrão e foi necessário que a incluíssemos por conta própria. Caso você deseje fazer isso desse modelo aqui descrito, é necessário que você copie o arquivo **total_mass_profiles.py** presente nesse diretório para dentro da pasta **path-where-installed-autolens/autogalaxy/profiles/mass_profiles**. O arquivo **total_mass_profiles.py** presente aqui é uma cópia do arquivo original presente na distribuição do Pyautolens, com a diferença de que foi adicionana uma nova classe chamada **MGE**, em que é possível utilizar a parametrização MGE para modelar sua lente. 

Ao copiar esta versão do documento **total_mass_profiles.py** para dentro do diretório **mass_profiles**, será pedido para que você substitua o arquivo lá presente. Caso não se sinta confortável com isso (ou por segurança queira manter salvo uma versão do código original) recomendamos que você faça uma cópia do arquivo original **total_mass_profiles.py** para outro diretório de sua preferência, ou ainda renomeie o arquivo original para **OFICIAL_total_mass_profiles.py**. 

Além de adiconar essa "nova versão" dos perfis de massa, é necessário informar ao arquivo **__init__.py** presente no diretório **/mass profiles** que uma nova classe está disponível. Isso é feito adicionando o nome **MGE** à função __.total_mass_profiles()__ presente dentro do arquivo **__init__.py** , como pode-se ver na imagem abaixo:

![Inclusão da classe **MGE** ao **__init__.py** presente no diretório **path-where-installed-autolens/autogalaxy/profiles/mass_profiles**](init-mass-profiles.png "**__init__.py**")


As demais recomendações de como se deve fazer uso do Pyautolens estão em sua própria documentação. Mas atenção, ao setar o caminho para o autolens_workspace você deve setar para o autolens_workspace presente neste diretório e não aquele disponível no github do James. Isso porque alguns arquivos de configuaração precisaram ser alterados para que a classe **MGE** fosse aceita. Em particular, foi necessária a inclusão da classe **MGE** no arquivo **radial_minimum.ini**, presente no diretório **autolens_workspace/config/grids**.

