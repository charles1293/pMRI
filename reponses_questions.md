# Reponses aux questions du TP pMRI

## Question 1

Avec R=2, l augmentation du bruit (sigma = 5, 10, 20, 30) degrade progressivement la qualite des images simulees.
Quand sigma est faible, les structures sont nettes et le contraste reste bon.
Quand sigma augmente, l image devient plus granuleuse, les contours se degradent et les details fins deviennent plus difficiles a distinguer.

## Question 2

En refaisant la simulation avec R=4, les acquisitions sont plus degradees qu avec R=2 pour un meme niveau de bruit.
La raison est le sous echantillonnage plus fort a R=4: il y a moins d information mesuree, donc plus de perte et plus de sensibilite au bruit.

## Question 3

La reconstruction confirme les observations precedentes:
le SNR diminue quand sigma augmente, et pour chaque sigma le SNR de R=2 est superieur a celui de R=4.
Visuellement, R=2 conserve mieux les structures anatomiques, alors que R=4 presente plus d artefacts et une perte de qualite plus rapide.

## Question 4

Avec la regularisation de Tikhonov (sigma fixe a 10 dans le script), une petite valeur de lambda peut ameliorer la stabilite, surtout pour R=4.
Quand lambda devient trop grand, la reconstruction est trop lissee et le SNR baisse.

Donc:
- lambda trop faible: regularisation insuffisante,
- lambda intermediaire: meilleur compromis,
- lambda trop fort: perte d information utile.

La conclusion est que la regularisation aide, mais le choix de lambda est determinent pour la qualite finale.
