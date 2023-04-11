## Introduction

Ce projet est un projet bac-à-sable technique sur lequel j'ai travaillé en 2020. C'est une réécriture quasi-complète d'un projet de Microsoft pour le challenge CodeSearchNet (moteur de recherche de code multi-langage à partir de requêtes textuelles, sujet qui depuis a été largement poussé plus loin par github/microsoft avec Copilot avec des capacités avancées de génération de code).
J'ai remis mon projet à jour pour cette présentation car je me suis aperçu que les API python utilisées en 2020 ont beaucoup évolué depuis et le code n'était plus du tout compatible avec les versions actuelles. Cependant, le code ne tournera pas si vous le lancez car il a besoin de tokenizers qu'il faut construire manuellement et qui demandent pas mal de temps de calcul et de librairies natives (pour les AST de langages).

## Points techniques remarquables

Mon but ici n'est pas de parler du fond (mes résultats n'étaient pas très intéressants) mais plutôt du code car à l'époque j'avais utilisé ce projet plus comme un bac-à-sable technique pour tester personnellement les points suivants:

- Projet complet Python/ML avec gestion des dépendances (poetry) et isolation dans un virtualenv, intégration dans VSCode avec utilisation d'outils classiques de dev: mypy, linters, license, tests (même si anecdotiques), etc...
  - [pyproject.toml](./pyproject.toml)
  - [license](./LICENSE)
- Utilisation des configurations au format générique HOCON qui permet de gérer des configurations complexes avec des imports, des variables, des références etc...
  - [Configuration générique](./conf/default.conf)
  - [Configuration spécifique](./conf/query_code_siamese_2020_02_15_14_00.conf)
- Exploration du typage fort en Python avec des types abstraits et les "newtypes" pour spécialiser des types simples

  - [Type abstraits](./codenets/recordable.py)
  - [Type génériques](./codenets/training_ctx.py#L205-L220)
  - [Newtypes](./codenets/codesearchnet/training_ctx.py#L49-L68)

- Evaluation de la compilation des types avc le moteur de compilation Mypy de Microsoft.
  - [mypy.ini](./mypy.ini)
- Etude de sauvegarde/restoration générique d'un contexte complet de projet IA pour une sauvegarde
  dans un point unique sur un cloud de type AWS ou un serveur orienté ML de type MLFlow avec lien vers des commits de code github par exemple.
  - [training context générique](./codenets/codesearchnet/training_ctx.py#L245)
  - [training context spécialisé](./codenets/codesearchnet/query_code_siamese/training_ctx.py)
- Evaluation de la complexité de réécriture d'un code Tensorflow vers du PyTorch et les librairies huggingface.
- Intégration avec WanDB/Tensorflow pour le suivi des entraînements.

et de manière plus spécifique:

- Etudier les résultats atteignables avec des transformers de petite taille sur un challenge de ce type
  et les résultats ont été très décevants cf. [readme](./README.md)
- Utilisation de tokenizers natifs Rust avec interface Python de Huggingface tokenizers (qui venaient d'être publiés en 2020): [tokenizer_recs.py](./codenets/codesearchnet/huggingface/tokenizer_recs.py#L102)
- Utilisation des parsers d'AST de langages (tree-sitter) pour améliorer les performances: [ast_buid.py](./codenets/codesearchnet/ast_build.py#L189)

## Conclusion

Je ne conseillerais pas spécialement la qualité de ce code qui est un peu compliqué selon mon point de vue mais je retiendrai les points suivants:

- L'utilisation des configurations HOCON est un réel pour tout projet informatique quel que soit le langage à mon avis car cela permet de gérer des configurations avec différents niveaux de généricité et permettant d'utiliser facilement des variables.
- le typage dans Python est fonctionnel et permet d'améliorer la robustesse globale du code et mypy semble être une solution efficace pour vérifier les types. Cependant, l'utilisation des types génériques et abstraits est assez lourde et les problèmes classiques de la programmation orienteé objet comme l'héritage multiple surgissent assez vite. L'utilisation des NewTypes reste anecdotique de mon point de vue car les opérations mathématiques sur ces types leur font perdre leur spécificité.
- la sauvegarde générique complète d'un projet ML est intéressante dans l'optique de backup et versioning de projets ML en associant l'intégralité des ressources: code, configuration, modèle, tokenizer etc...
