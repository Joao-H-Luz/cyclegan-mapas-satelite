# CycleGAN: mapas históricos ↔ imagens de satélite

Trabalho da disciplina de Aprendizado Não-Supervisionado (PUCRS, Prof. Luan Garcia). Implementamos e treinamos do zero uma CycleGAN para tradução de imagens **não pareada** e analisamos criticamente os resultados.

Para reproduzir o treino, veja [COMO_RODAR.md](COMO_RODAR.md).

## Tradução escolhida e motivação

**Mapa histórico (A) ↔ ortoimagem de satélite (B).**

- **A → B:** dado um mapa antigo, gerar como a região "aparece de cima".
- **B → A:** dado uma foto aérea, gerar um mapa no estilo histórico.

Os dois domínios descrevem o mesmo tipo de coisa (ruas, quadras, rios, ferrovias), mas com aparências muito diferentes. Isso torna o problema interessante para uma CycleGAN, e também difícil: o mapa é uma abstração, com traço, cor chapada e texto, e a foto tem textura e informação que o mapa não carrega (telhados, árvores, sombras). Por isso a tradução não é bijetiva, e esperamos ver aí tanto sucessos quanto falhas informativas para a análise crítica. O par mapa↔aéreo também é o clássico do pix2pix (Isola et al., 2017), mas aqui os dados **não são pareados**.

## Domínios e dataset

Todas as imagens são JPEG 256×256 RGB.

| Domínio | Conteúdo | Treino | Teste |
|---|---|---|---|
| A | recortes de mapas históricos (David Rumsey Map Collection): cadastrais, topográficos, rodoviários, muitos com texto | 6099 | 1056 |
| B | ortoimagens aéreas do USGS de cidades dos EUA, nos zooms 14–16 | 6586 | 947 |

As cidades do conjunto de teste são **diferentes** das de treino, para que o teste meça generalização e não memorização.

![Exemplos dos dois domínios](figuras/dataset_exemplos.jpg)

*Seis exemplos de A (em cima) e de B (embaixo). Gerada com `uv run python analyze.py dataset --data data`.*

### Origem dos dados

- **Domínio A:** David Rumsey Map Collection.
- **Domínio B:** USGS National Map (ortoimagens).
- **TODO (grupo):** descrever como os dados foram coletados (ferramenta ou script, critério de escolha das cidades e dos mapas, como foram feitos os recortes 256×256, filtros e limpeza, licença de uso).

Os dados (≈ 553 MB) não ficam no repositório.

## Implementação

Seguimos a CycleGAN original (Zhu et al., 2017). Tudo está em PyTorch, sem dependências além de PIL e matplotlib.

### Arquitetura

- **Geradores `G_AB` e `G_BA`: ResNet de 9 blocos.** `c7s1-64, d128, d256, 9×R256, u128, u64, c7s1-3`, com ReflectionPad, InstanceNorm e ReLU, e saída `tanh` em [-1, 1]. Para 256×256 a literatura usa 9 blocos; o resnet preserva bem a estrutura espacial (o que é desejável aqui, pois o traçado de ruas deve ser mantido), e a InstanceNorm funciona melhor que a BatchNorm em tradução de estilo com batch 1.
- **Discriminadores `D_A` e `D_B`: PatchGAN 70×70.** `C64 - C128 - C256 - C512 - conv 1 canal`, LeakyReLU 0.2, InstanceNorm (sem norma na primeira camada). Para entrada de 256 px, a saída é um mapa 30×30 de decisões real/falso, cada uma olhando um trecho de 70×70 px. Julgar texturas locais é o que importa para estilo, e tem poucos parâmetros e funciona com qualquer tamanho de imagem.
- Inicialização dos pesos: normal com desvio 0,02.

### Perdas

- **Adversarial LSGAN** (erro quadrático contra 1/0, sem sigmoid). Gradientes mais estáveis que a entropia cruzada e sem saturação; é a escolha do paper original. A perda do D é multiplicada por 0,5 para que ele aprenda mais devagar que o G.
- **Consistência cíclica L1, λ = 10.** `a → G_AB → G_BA ≈ a` (e o análogo para B). É o que permite treinar sem pares: obriga a tradução a manter o conteúdo. L1 gera menos borrão que L2.
- **Identity desligada no treino principal** (`λ_id = 0`). Nos mapas, a identity forçaria `G_AB(b) ≈ b`, o que não é natural aqui (um gerador de B a partir de B). Ela serve sobretudo para preservar cor, e o domínio de satélite já tem paleta própria. Vamos testá-la em uma **ablação** curta (`λ_id = 0,5`) para medir o efeito.
- **Image pool** de 50 imagens falsas para o D, que estabiliza o treino (Shrivastava et al., 2017).

### Otimização

- Adam, `lr = 2e-4`, betas `(0.5, 0.999)`, batch 1.
- **Agendamento de LR fixo em 100 épocas:** o LR fica constante até a época 50 e depois cai linearmente, chegando perto de zero na época 100. O valor de cada época é calculado direto dos argumentos (`epoca`, `epochs`, `decay-start`), sem estado de scheduler. Assim o treino pode ser interrompido e retomado a qualquer hora sem alterar o agendamento.
- **AMP fp16** ligado por padrão para caber em 6 GB de VRAM (RTX 2060). As perdas são calculadas em float32.
- Uma época = `max(|A|, |B|)` = 6586 iterações. O lado A segue a ordem embaralhada e B é sorteado aleatoriamente, o que mantém o treino não pareado.

### Por que não há data augmentation

Não usamos flip nem outras transformações. O flip horizontal espelharia o **texto dos mapas**, criando exemplos que não existem no domínio. Também não há recorte aleatório: as imagens já têm 256×256 e os recortes foram feitos na coleta. Com ~6 mil imagens por domínio e PatchGAN com poucos parâmetros, o risco de sobreajuste é moderado, e deixamos a avaliação dele para a análise (curvas e casos de teste).

## Estrutura do repositório

```
README.md           este arquivo
COMO_RODAR.md       passo a passo do treino no PC com GPU (Windows)
pyproject.toml      dependências (uv)
models.py           geradores, discriminadores, E/S de imagem
train.py            treino, checkpoints e retomada
analyze.py          figuras e análises a partir de runs/
avaliar.py          inferência no conjunto de teste, casos bons/ruins, métricas
                    e comparação da ablação (identity loss)
monitor.py          acompanha o treino pelo celular (ntfy.sh), com alerta de travamento
tests/              teste de fumaça (CPU)
figuras/            figuras gerais (dataset)
runs/<run>/         saída de cada treino: config.json, perdas.csv, epocas.csv,
                    amostras/epoca_XXX.jpg, curvas_*.png, evolucao_*.jpg
data/               (fora do git) trainA/ trainB/ testA/ testB/
```

Os pesos (`*.pt`) ficam fora do git, exceto o snapshot final dos geradores, `runs/principal/G_epoca_100.pt` (44 MB), que permite rodar o `avaliar.py` sem retreinar.

Os slides da apresentação estão em `apresentacao.html` (abre offline no navegador; ← → navegam, N mostra as notas, F tela cheia).

## Resultados e análise

Treino principal: 100 épocas, uma única execução. Os arquivos estão em `runs/principal/`. Onde o texto diz _pendente_, a métrica ou o teste ainda não foi rodado; o comando que gera cada um está indicado.

### Treino

- **Parâmetros** (`runs/principal/config.json`): 100 épocas, LR constante até a época 50 e depois decaindo (`decay_start = 50`), λ_cyc = 10, λ_id = 0, lr = 2e-4, batch 1, AMP fp16, seed 42.
- **Hardware:** RTX 2060 6 GB. Cada época tem 6586 iterações.
- **Tempo por época:** 24,1 min em média (~4,6 it/s). Foi de cerca de 22,9 min até a época 37 e de cerca de 24,8 min depois, quando outros programas passaram a disputar a CPU. O gargalo era a CPU (uma thread no limite), não a GPU.
- **Tempo total:** 40,1 h.
- **VRAM de pico:** 5,0 GB.
- O treino foi pausado e retomado várias vezes (inclusive após uma queda de energia) sem alterar o agendamento do LR, porque o LR de cada época é calculado direto do número da época.

As figuras `curvas_adversarial.png` e `curvas_ciclica.png` ainda não foram geradas (`uv run python analyze.py curvas principal` as gera a partir de `runs/principal/`). Os valores abaixo vêm de `runs/principal/epocas.csv` (média por época).

| Época | G_adv A→B | G_adv B→A | cyc_A | cyc_B | D_A | D_B |
|---|---|---|---|---|---|---|
| 1 | 0,452 | 0,553 | 0,231 | 0,209 | 0,153 | 0,196 |
| 10 | 0,618 | 0,702 | 0,137 | 0,128 | 0,069 | 0,068 |
| 25 | 0,726 | 0,700 | 0,124 | 0,114 | 0,069 | 0,045 |
| 50 | 0,775 | 0,784 | 0,111 | 0,093 | 0,050 | 0,036 |
| 75 | 0,867 | 0,799 | 0,099 | 0,080 | 0,044 | 0,021 |
| 100 | 0,961 | 0,814 | 0,092 | 0,078 | 0,049 | 0,011 |

**Curvas adversariais (LSGAN, erro quadrático):**

- D_B (satélite) cai de 0,20 para 0,01 enquanto G_adv A→B sobe de 0,45 para 0,96, o treino inteiro, inclusive na fase de LR decrescente. O discriminador de satélite distingue com facilidade as fotos geradas.
- No outro sentido o equilíbrio é melhor: G_adv B→A estabiliza perto de 0,8 e D_A perto de 0,05.
- Os picos de G_adv B→A nas épocas 61–65 e 83–84 são discutidos na falha 4 (instabilidades).

**Curva da perda cíclica (L1, escala [-1, 1]):**

- As duas caem cerca de 60%: cyc_A de 0,23 para 0,09 e cyc_B de 0,21 para 0,08.
- A queda é rápida nas primeiras 10 épocas e depois lenta e contínua, também na fase de LR decrescente.
- Uma perda cíclica baixa não garante uma tradução boa (ver falha 3).

### Evolução por época

![Evolução A→B](runs/principal/evolucao_A2B.jpg)

![Evolução B→A](runs/principal/evolucao_B2A.jpg)

São 4 + 4 imagens fixas do **conjunto de teste** (nunca vistas no treino), traduzidas nas épocas 1, 5, 10, 25, 50, 75 e 100, mais a reconstrução da época 100. As amostras de todas as épocas ficam em `runs/principal/amostras/`.

- **A→B:** na época 1 tudo é verde-escuro com manchas; na época 10, tom marrom-oliva; da época 50 em diante, cores mais calmas e textura de satélite. A estrutura (costa, lagoa, quadras, rios) aparece cedo e quase não muda; o que evolui é a cor e a textura. A reconstrução da época 100 é fiel nos quatro casos.
- **B→A:** o estilo de mapa (papel creme, tinta, hachura) aparece já na época 10. Na época 5 as quatro saídas têm faixas horizontais coloridas, que somem depois. Na época 100 surgem manchas rosa-magenta que não existiam nas épocas 50 e 75.

### Casos bons e ruins

Dois bons e dois ruins em cada sentido, todos do conjunto de teste (a numeração segue a ordem das imagens fixas em `runs/principal/config.json`). A escolha é visual, sobre as 4 + 4 imagens fixas. A seleção sobre o teste inteiro está _pendente_: `uv run python avaliar.py inferir principal` e depois `uv run python avaliar.py casos principal`.

**Mapa → satélite (A→B)**

| Imagem | Julgamento | Por quê |
|---|---|---|
| 1 | Bom | Costa e lagoa preservadas; o mar vira textura de mata. |
| 2 | Ruim | Lotes viram blocos escuros; os números são copiados. |
| 3 | Ruim | Imagem lavada, relevo irreal; "CANAL" continua escrito. |
| 4 | Bom | Rios e estradas preservados; parece foto aérea. |

**Satélite → mapa (B→A)** (Providence, Baltimore em dois zooms e Sacramento, cidades fora do treino)

| Imagem | Julgamento | Por quê |
|---|---|---|
| 1 (Providence) | Ruim | Manchas rosa sem correspondência e traço pesado. |
| 2 (Baltimore, z16) | Bom | Quadras e prédios preservados, estilo convincente. |
| 3 (Baltimore, z14) | Bom | Malha fina em estilo de gravura, pouco ruído de cor. |
| 4 (Sacramento) | Ruim | As árvores somem: sobra papel e pseudo-texto. |

O pior caso é o bairro muito arborizado: copa de árvore não tem equivalente em mapa, então o gerador entrega papel vazio com rabiscos.

### Análise crítica

**Falha 1: o texto do mapa não some, é copiado ou inventado.**

- *Evidência:* em A→B, "CANAL", "DOUGLAS", "LEMBERG" e os números dos lotes "404" e "405" continuam como letras claras sobre o satélite gerado. Em B→A surgem rótulos e uma escrita cursiva sem sentido, sobretudo no bairro arborizado.
- *Hipótese:* satélite não tem equivalente para texto. Apagá-lo custaria perda cíclica, e o PatchGAN 70×70 julga textura local, então tolera letras pequenas. No sentido inverso, D_A aprendeu que mapas têm rótulos, e o G_BA desenha algo com cara de texto para enganá-lo. É a CycleGAN fazendo exatamente o que a perda pede: a perda cíclica exige reconstruir o texto, e nada na perda adversarial pune com força letras pequenas.
- *Como testar:* contar trechos com texto nas saídas em todo o conjunto de teste; treinar com recortes sem texto, ou mascarar o texto na perda cíclica, e comparar. Ainda não feito.

**Falha 2: mapa → satélite é o sentido mais difícil (D_B domina).**

- *Evidência nas curvas:* D_B cai de 0,20 para 0,01 e G_adv A→B sobe de 0,45 para 0,96 até o fim. No outro sentido, D_A fica perto de 0,05 e G_adv B→A estabiliza perto de 0,8.
- *Evidência nas imagens:* as traduções A→B saem lavadas e acinzentadas, com textura repetitiva: o mar vira copa de árvore e o relevo vira cristas pálidas.
- *Hipótese:* assimetria de informação. Ir de foto para mapa é descartar detalhe; ir de mapa para foto é inventar telhados, copas e sombras que o mapa não registra. O gerador fica numa textura média, que o D_B reconhece com facilidade.
- A perda do gerador subir durante todo o treino, com o discriminador indo a quase zero, é o sinal clássico de discriminador forte demais. Isso não derrubou o treino (a perda cíclica seguiu caindo e não houve colapso total), mas limita o realismo de A→B. Possíveis ajustes: reduzir o aprendizado do D_B, usar data augmentation no D ou regularização como gradient penalty.

**Falha 3: reconstruções quase perfeitas a partir de traduções ruins (hipótese de esteganografia).**

- *Evidência:* nos 8 casos de teste a reconstrução é fiel, mesmo quando a tradução falha. O bairro arborizado vira quase papel vazio no mapa, mas a reconstrução devolve as copas das árvores.
- *Hipótese:* esteganografia (Chu et al., 2017). O gerador esconde a imagem original em sinais de baixa amplitude, quase invisíveis, e o outro gerador os decodifica. A perda cíclica cai sem a tradução ficar fiel. Uma tradução que descarta as copas não deveria permitir reconstruí-las; se a reconstrução volta, a informação passou por algum canal escondido.
- *Como testar:* somar um ruído pequeno (σ = 0,02 e 0,05) à tradução antes do segundo gerador. Se a reconstrução piorar muito mais do que o ruído explica, há informação escondida. O teste está implementado em `avaliar.py`.
- *Resultado:* _pendente_. Gerar com `uv run python avaliar.py esteganografia principal`.

**Falha 4: instabilidades.** O treino nunca colapsou: a perda cíclica caiu em todas as fases e nenhuma perda virou NaN. Houve oscilações pontuais, das quais ele se recuperou.

| Quando | O que aconteceu | Hipótese |
|---|---|---|
| Época 5, satélite → mapa | As 4 saídas de teste com faixas horizontais coloridas; somem na época 10. | Oscilação do começo do treino, com o D_A ainda instável. |
| Épocas 61–65 e 83–84 | G_adv B→A sobe a 0,88–0,98 e D_A cai a cerca de 0,002; na época seguinte voltam a ~0,8 e ~0,05. | O D_A vence por algumas épocas; o equilíbrio se refaz sem intervenção. |
| Época 100, satélite → mapa | Manchas rosa-magenta fortes em 3 das 4 imagens, ausentes nas épocas 50 e 75. | Possível imitação do rosa de prédios em mapas antigos, ou artefato tardio; confirmar no teste inteiro. |

Nas épocas 83 e 84 a média de D_A chegou a 0,002, ou seja, o discriminador de mapas acertava quase tudo e o gerador satélite→mapa perdia; na época 85 o equilíbrio voltou. Sobre o rosa da última época: mapas de seguro antigos pintam prédios de rosa, então pode ser um estilo do domínio A, mas com só 4 imagens não dá para afirmar.

### Limitações

- **Avaliação:** a análise visual usa 4 + 4 imagens fixas de teste. As métricas no conjunto de teste inteiro (cíclica L1, PSNR, SSIM) estão _pendentes_: `uv run python avaliar.py inferir principal`.
- **Experimentos:** uma única execução, com seed 42 e batch 1, sem busca de hiperparâmetros. A ablação com identity loss não foi treinada.
- **Dados:** domínios desalinhados (mapas de várias regiões e escalas contra cidades dos EUA em zooms 14–16). Parte das diferenças pode vir disso, não do modelo; por exemplo, o mar virando textura de mata.
- As conclusões sobre casos bons e ruins vêm de 8 imagens. Para afirmar padrões gerais seria preciso olhar o conjunto de teste inteiro e ter métricas.

### Conclusão

A CycleGAN aprendeu o estilo e a estrutura, mas não aprende a inventar o que falta.

- **Satélite → mapa funciona melhor:** descartar detalhe é fácil, e ruas e quadras viram traço de mapa convincente.
- **Mapa → satélite fica limitado:** é preciso inventar textura que o mapa não tem, e o discriminador de satélite vence.
- **Perda cíclica baixa não basta:** reconstruções perfeitas a partir de traduções ruins sugerem informação escondida (a confirmar com o teste de esteganografia).

Próximos passos: ablação com identity loss, máscara de texto e mapas e satélites da mesma região e escala.

## Autores

- João Henrique da Luz
- Vittor Brits
- Marcelo

## Referências

- Zhu, Park, Isola, Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017.
- Isola, Zhu, Zhou, Efros. *Image-to-Image Translation with Conditional Adversarial Networks (pix2pix).* CVPR 2017.
- Mao et al. *Least Squares Generative Adversarial Networks.* ICCV 2017.
- Shrivastava et al. *Learning from Simulated and Unsupervised Images through Adversarial Training.* CVPR 2017 (image pool).
- Chu, Zhmoginov, Sandler. *CycleGAN, a Master of Steganography.* 2017.
- Dados: [David Rumsey Map Collection](https://www.davidrumsey.com/) e [USGS National Map](https://www.usgs.gov/programs/national-geospatial-program/national-map).
