# [GAN논문 1팀] 1주차. GAN, pix2pix, and CycleGAN

---

# 1. Introduction

---

- 첫주차에는 Generative model 쪽에서 오랜 시간 SOTA로 자리매김했던 GAN의 논문과 수학적 background, 그리고 generative model의 background에 대하여 살펴보았다.
- 딥러닝의 다른 분야보다 Generative model쪽은 loss function이 상대적으로 중요하다보니, 수학적 background가 많이 필요하다.
- 이를 위해 간단하게 background를 정의하겠다.

## 1.1 Background

---

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled.png)

- 먼저, 용어적인 정의는 알고 있다는 전제하에 일반적으로 확률 공간을 생각하면 **Sample Space**를 실수 $\sigma$-space로 대응시키는 **Random Variable**과 Random Variable을 0~1사이의 확률값 probability로 대응시키는 measure function인 **Distribution function**을 생각할 수 있다.
- 이 때, 이러한 두개의 measure function을 통한 대응 방식은 우리가 생각하는 generative 모델에서도 대응시킬 수 있다.

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%201.png)

- 어떤 $R^n$차원인 pixel 차원에 존재하는 우리가 가진 training image dataset은 각각 probability를 가지고 어떤 확률분포를 가질 것이다. 이를 우리는 이 둘사이의 measure function, 즉 distribution function을 우리는 $p_{data}(x)$라고 정의할 수 있다.
- 또한, generative model의 task는 결국 training data와 유사한 이미지를 만들어내는 것인데, 여기서 이미지가 유사하다는 것을 수학적으로 표현하자면 “**************************************************************************확률분포가 서로 비슷하다”**************************************************************************라고 표현한다.
- 즉, Generative model에서는 뒤에서도 얘기하겠지만, 결과적인 목적은 이 $p_{data}(x)$라는 함수가 어떤 아이인지 backpropagation을 통해 parameterize하고 싶은 것이다. 하지만, 우리는 이것을 직접적으로 구하는 것이 불가능하다.
- 따라서, 이를 가능하게 하기 위해 Generagtive model들은  $p_{data}(x)$를 얻기위해 prior distribution이자 latent vector인 z로부터 x를 얻어내는 방식으로 **G(z)**(혹은 Decoder(z))를 이  $p_{data}(x)$를 생성하는 함수로서  $p_{data}(x)$를 parameterize한다.
- 이렇게 생성된 G(z)가  $p_{data}(x)$로 backpropagation을 통해 optimize를 하게 되면 앞서 말한 유사한 이미지를 생성하게 되는 것이고, 여기서 generative model에서의 확장성은 결과적으로 **그렇다면 이 두 확률 분포를 근사시키기 위해 어떤 divergence rule을 가지고 모델링하여 loss function을 어떤 아이로 결정할 것인가**에 있다.

# 2. Problem Formulation & Solution

---

- [ ]  **VAE**
- 당시, 기존에 존재하던 generative 모델인 VAE는 몇가지 문제점이 존재했다.
1. KL divergence를 사용하다보니, VAE의 encoder단계에서 encoder의 output을 gaussian distribution으로 추정하고, z를 normal distribution으로 추정하는게 필연적이였다.
2. 그러다보니, 그과정에서 결과적으로 나타나는 loss function은 반드시 MSE loss, 즉 L2 distance를 포함하게 되고 이는 픽셀상으로 blurry한 이미지를 출력하게 된다.

⇒ **즉, KL divergence를 바탕으로 두 확률분포의 거리의 척도를 계산하다보니 loss function이 모델링할 수 있는 dataset이 제한된다는 것이다.**

<aside>
💡 **Solution:** 이를 위해 GAN 논문에서는 **JSD에 기반한 Min-Max game theory를 적용하여 Adversarial model을 제안**한다. 이를 통해 **VAE처럼 z와 x의 분포를 explicit하게 정의하고 loss를 학습하는 것이 아니라**, **x의 분포를 explicit하게 정의하지 않고, 주어진 x의 분포자체로 G(z)가 수렴해가는 방식으로 훈련할 수 있음**을 증명한다.

</aside>

# 3. Loss function

---

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%202.png)

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%203.png)

- 일반적으로 GAN을 학습할 때에는 따라서 두 network를 번갈하가면서 학습시키게 된다.
- 이를 위해 위 그림에서 두 network G와 D는 하나의 loss function에 대해 각각 다른 방향으로 update하며, 본 논문에서는 이러한 Adversarial Network의 loss function으로 다음과 같이 정의한다.

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%204.png)

- 이 loss function은 결과적으로 고정된 G에 대해 optimize하는 D의 paramerter값이 미분을 통해

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%205.png)

일 때 성립하는 것을 알 수가 있는데, 이 값을 넣어주게 되면 **결과적으로 앞서 제안한 loss function이 JSD를 따른다**는 것을 본 논문에서는 증명하는 과정을 보여주었다. 즉, 앞선 loss function을 optimize한다는 것은 결과적으로 $JSD(p_{data}||p_g)$를 최적화하는 과정을 나타낸다는 것이다.

- 때문에, 본 loss function이 두 확률분포를 잘 근사할 수 있다고 설명한다. 이 부분에서 그렇다면 JSD가 아닌 다른 divergence rule을 바탕으로 modeling한다면 또다른 generative model을 후속연구로 할 수 있지 않을까하는 생각이 들었다.

# 4. Variation of GAN

---

## 4.1 Pix2Pix(2018)

---

- [ ]  **Problem Formulation**

---

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%206.png)

- PIx2Pix는 기존의 vanila GAN과는 task에 있어서 조금 차이가 있는데, Pix2Pix는 conditional GAN, 즉 input으로 x image와 y image를 함께 주고, 넣어준 **input image x를 image y와 비슷한 형태로 그림을 그리는 task에 대한 논문**이다.
- 이를 x의 domain에서 y의 domain으로 옮겨주는 것이라고 표현하기도 한다.

- [ ]  **Solution**

---

- 이를 위해 GAN과 마찬가지로, Generator-Discriminator 두 가지를 가지고 전체적인 네트워크를 구성한다.
- 이 때, Generator와 Discriminator의 구체적인 모델은 일반적인 MLP가 아닌 task에 맞게 튜닝을 해준다.
1. **Generator**

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%207.png)

- Generator는 image-to-image translation 특성상 input 차원과 output 차원이 같아야하기 때문에, AutoEncoder와 매우 유사한 형태의 architecture로 설정한다.
- 다만, 여기서 성능을 더해주기 위해, Encoder-Decoder형태의 network에서 성능이 좋다고 잘 알려진 **U-Net의 구조**를 따와서 skip-connection을 추가하여 input image로부터 fake image y hat을 생성하게 된다.
1. **************************Discriminator**************************
- Discriminator의 경우 generate한 이미지의 해상도를 높이기 위해 기존에 존재하던 PatchGAN의 구조를 따온다.
- 즉, 이미지 전체를 보고 fake/real을 판별하는 것이 아니라, 전체 이미지를 patch로 NxN으로 쪼개어 각각의 patch별로 fake/real을 판별하는 것이다.
1. ********Loss function********

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%208.png)

 where,

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%209.png)

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%2010.png)

- loss function으로 이를 위해, 맨 위의 식을 활용한다. 왼쪽의 term은 기존의 GAN과 똑같이 min-max game에 기반한 loss인데 input이 2가지이기 때문에 위와 같이 써준 것이다.
- 그리고 오른쪽의 term이 새로 loss로 추가되었는데, 이는 generate한 이미지가 실제 이미지와 가깝게 하도록하는 term이다. 즉, 쉽게 말해 supervised learning의 형태 또한 동시에 가지고 있기 때문에, 이를 위해 distance에 기반한 loss term을 추가해준 것이다.
- 여기서 본 논문의 저자는 L2 loss는 blurry하게 하는 효과가 있으니 이를 방지하기 위해 L1 loss를 적용했다고 한다.

## 4.2 CycleGAN

---

- [ ]  **Problem Formulation**

---

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%2011.png)

- Pix2Pix의 논문은 앞서서 loss상에서 살펴봤듯이 반드시 x와 y를 pairing하여 함께 input으로 넣어줘야 학습이 일어난다.
- 이러한 pairing의 가장 큰 문제점은, dataset을 구축하는데에 매우 큰 어려움이 있다는 것이다.
- 이를 위해, 본 논문에서는 pairing 없이더 image-to-image translation을 할 수 있는 cycle기반의 학습을 제안한다.

- [ ]  **Solution**

---

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%2012.png)

- 우리가 어떤 번역기가 정말 해석을 잘 한다면, 영어→프랑스어로 번역하고 다시 이를 영어로 번역해도 같은 문장이 나올 것이다.
- 이러한 inspiration을 그대로 적용한 것이 바로 CycleGAN이다.
- 본 논문에서는 앞서 말한 번역기 예시를 그대로 사용하면서 본 논문의 architecture를 설명한다.
- 즉, **Generator G와 그의 역함수 F에 대하여, $F(G(x))\sim x$ 이고,$G(F(y))\sim y$라면, 우리가 생성한 generator는 잘 동작하고 있다고 생각하는 것이 CycleGAN의 철학**인 것이다.
- 이를 위해 본 논문은 아래와 같은 Loss를 제안한다.

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%2013.png)

,where

![Untitled](%5BGAN%E1%84%82%E1%85%A9%E1%86%AB%E1%84%86%E1%85%AE%E1%86%AB%201%E1%84%90%E1%85%B5%E1%86%B7%5D%201%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20GAN,%20pix2pix,%20and%20CycleGAN%20f39d882d99554642af652508c87776be/Untitled%2014.png)

- 위에서 $**L_{GAN}$은 두 generator G,F와 각각에 대한 discriminator에 대한 GAN Loss를 나타내고, $L_{cycle}$은 pix2pix의 L1 distance와 같은 철학으로 supervised learning의 형태로 F와 G가 서로 역함수관계에 있도록 학습하게 하는 loss function**이다.

# 4. Discussion

---

- [ ]  **random latent vector z의 의미**
- GAN에서 generator의 input으로 넣어주는 latent vector는 일반적으로 normal distribution에서 random하게 sampling하여 input으로 넣어주게 된다. 이에 대해 두 가지 해석이 가능하다.
1. 어떤 $R^k$차원의 latent vector를 sample space로 볼때 generator는 얘를 실수공간 $R^N$으로 mapping해주는 random variable이다. 즉 어떤 저차원의 의미를 고차원 pixel 차원으로 확장시켜주는 역할이 바로 generator인 것이다.
2. 또는 비슷한 의미겠지만, z는 manifold space상의 임의의 벡터이니 z를 random한 값으로 설정해도 어차피 manifold space상의 특징을 갖고 있는 임의의 점일 것이다. 그렇다면 generator의 역할은 manifold상의 임의의 벡터를 $R^N$차원으로 복원시키는 것으로 생각할 수 있다.

- [ ]  **Alternative learning방식의 이론과의 괴리?**

 **실제:** $G_{n+1}\leftarrow argmin\ V(D^*_{G_n},G_n)$ 

 **이론:** $G \leftarrow argminV(D,G)$

- GAN을 학습할 때에는, 앞서 설명했듯이 D와 G가 번갈아가며 학습되기 때문에, 매 epoch마다 D는 다른 G에 대하여 학습될 것이다.
- 여기서 앞서 C(G)에서는 D가 고정된 G에 대하여 학습되는 것을 통해 JSD를 따르는 것을 이론적으로 증명했는데, 이러한 학습방식은 이론과 괴리가 존재하지 않느냐에 대한 논의를 가졌다.
- 왜냐하면, 만약 고정된 G에 대해 정의된 D, 즉 $V(p_g,D)$가 convex하지 않은 형태에서 optimize된다면, 학습이 잘못된 방향으로 일어날 수 있고 이를 속이기 위해 생성된 G는 또다시 잘못된 방향으로 이끌어질 수도 있다.
- 즉 완벽하게 최적화가 일어나지 않을 수 있다는 논의가 이루어졌었고, 결과적으로 추후에 이러한 문제점을 해결하기 위해 후속논문들이 출시된 것을 확인할 수 있었다.
- 이러한 문제점은 실제로 Nash equilibrium, Non-Convergence등의 용어로 불린다.