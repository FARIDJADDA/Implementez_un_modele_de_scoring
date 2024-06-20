# Mission - Élaborez le modèle de scoring
![baniere](https://user.oc-static.com/upload/2023/10/10/16969371520395_Section%20mission.png)

Cette mission suit un scénario de projet professionnel.

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
 ![baniere_entreprise](https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png)

 L’entreprise souhaite **mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité** qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un **algorithme de classification** en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

**Mission :**

1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

2. Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

3. Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.

4. Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

# Qu'allons-nous apprendre dans ce projet ?
Ce projet va reprendre un contexte qui vous est désormais familier : un problème de modélisation supervisée via une classification binaire. Toutefois, vous allez vous concentrer sur l’apprentissage d’une nouvelle compétence centrale : **la gestion du cycle de vie d’un modèle de ML.** 


En effet, après une très brève analyse exploratoire, vous allez entamer vos travaux de modélisation en utilisant un outil de versionning des modèles comme MLFlow, un outil qui a pris une place centrale dans la communauté Data Science ces dernières années. 


Après avoir sécurisé votre modèle le plus performant et avoir rendu transparent l’impact des features sur votre modèle, vous allez le mettre en production dans le cloud. Pour ce faire, vous allez consolider davantage vos acquis du projet précédent en ce qui concerne l’utilisation des APIs. Vous allez concevoir votre propre API d’ailleurs, en utilisant des librairies comme FastAPI ou Flask. Enfin, vous allez gérer le cycle de vie du modèle via du CI/CD et l’étude du drift entre autres. 


## Pourquoi ces compétences sont-elles importantes pour votre carrière ?
 

En plus d’être capable de résoudre un problème métier en sécurisant les bonnes features et le bon modèle de ML, il est de plus en plus attendu aujourd’hui qu’un Data Scientist peut déployer son modèle dans le Cloud pour l’exposer au client de son équipe. 


Effectivement, si un modèle de ML réussit à avoir un impact métier fort dans votre entreprise, de plus en plus de clients voudront l’utiliser. Ainsi, vous aurez très rapidement besoin d’automatiser l’accès aux prédictions pour que vos clients n’aient plus à vous solliciter pour obtenir les résultats du modèle. Ce processus fait partie de ce que l’on appelle le déploiement à l'échelle et représente une étape clé du cycle de vie d’un modèle de ML. 


L’ensemble des pratiques et raisonnements qui impactent le cycle de vie d’un modèle est souvent groupé sous le terme MLOps (Machine Learning Operations) qui se démocratise de plus en plus dans la communauté Data Science et dans les prérequis des fiches de postes. En somme, il ne suffit plus aujourd'hui d’avoir la capacité de créer des modèles, il faut également avoir les compétences techniques pour les rendre accessibles à l'échelle et de les faire vivre dans le temps, faute de quoi votre modèle cessera d’avoir l’impact métier positif qu’il a eu initialement dans votre entreprise.