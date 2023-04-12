## Introduction

Ce projet est un projet bac-à-sable technique sur lequel j'ai travaillé en 2020 et que j'utilisais pour tester certains points techniques à titre personnel. C'est une réécriture quasi-complète d'un projet de Microsoft pour le challenge CodeSearchNet (moteur de recherche de code multi-langage à partir de requêtes textuelles, sujet qui depuis a été largement poussé plus loin par github/microsoft avec Copilot avec des capacités avancées de génération de code).
J'ai remis ce projet à jour pour cette présentation car je me suis aperçu que les API python ont beaucoup évolué depuis 2020 et le code n'était plus du tout compatible avec les versions actuelles. Cependant, le code ne tournera pas si vous le lancez car il a besoin de tokenizers qu'il faut construire manuellement et qui demandent pas mal de temps de calcul et de librairies natives (pour les AST de langages).

## Points techniques remarquables

Mon but ici n'est pas de parler du fond ML/IA (mes résultats n'étaient pas très intéressants) mais plutôt du code et plus spécifiquement les points suivants:

- Projet complet Python/ML avec gestion des dépendances (poetry), isolation dans un virtualenv, intégration dans VSCode (qui devenait le standard de dev en 2020) avec utilisation d'extensions: mypy, linters, license, tests (même si anecdotiques), etc...

  - [pyproject.toml](./pyproject.toml)
  - et pour info, la [license](./LICENSE)

- Utilisation des configurations au format générique HOCON qui permet de gérer des configurations complexes avec des imports, des variables, des références etc...

  - [Configuration générique](./conf/default.conf)
  - [Configuration spécifique](./conf/query_code_siamese_2020_02_15_14_00.conf)

- Exploration des limites du typage fort en Python avec des types génériques abstraits (pour tenter de simuler l'équivalent des "typeclasses" qu'on trouve dans les langages fonctionnels comme Haskell/Scala) et les "newtypes" pour "spécialiser" des types simples

  - [Type abstraits](./codenets/recordable.py#L22)
  - [Type génériques](./codenets/codesearchnet/training_ctx.py#L205-L220)
  - [Newtypes](./codenets/codesearchnet/training_ctx.py#L49-L68)

- Evaluation de la compilation des types avc le moteur de compilation Mypy de Microsoft intégré dans VS Code.

  - [mypy.ini](./mypy.ini)

- Etude de sauvegarde/restoration générique d'un contexte complet de projet IA (configuration + commit + modèle + tokenizer + dataset + etc...) pour une sauvegarde dans un point unique (sur un cloud de type AWS ou un serveur orienté ML de type MLFlow par exemple).

  - [Recordable générique](./codenets/recordable.py#L22)
  - [Recordable spécialisé configuration HOCON](./codenets/recordable.py#L113)
  - [Recordable spécialisé modèle/tokenizer TorchModule](./codenets/recordable.py#L248)
  - [training context générique](./codenets/codesearchnet/training_ctx.py#L245)
  - [training context spécialisé sur un modèle spécifique](./codenets/codesearchnet/query_code_siamese/training_ctx.py#L40)

- Evaluation de la complexité de réécriture d'un code Tensorflow vers du PyTorch et les librairies huggingface.

- Intégration avec WanDB/Tensorflow pour le suivi des entraînements.

et de manière plus anecdotique:

- Etudier les résultats atteignables avec des transformers de petite taille sur un challenge de ce type
  et les résultats ont été très décevants cf.

  - [README.md](./README.md)

- Utilisation de tokenizers natifs Rust avec interface Python de Huggingface tokenizers (qui venaient d'être publiés en 2020):

  - [tokenizer_recs.py](./codenets/codesearchnet/huggingface/tokenizer_recs.py#L102)

- Utilisation des parsers d'AST de langages (tree-sitter) pour améliorer les performances des modèles à base de transformers (je n'ai pas réussi à pousser les expérimentations très loin par manque de ressources GPU)
  - [ast_build.py](./codenets/codesearchnet/ast_build.py#L189)

## Conclusion

Au final, je retiendrai les points suivants:

- L'utilisation des configurations HOCON est intéressante pour tout projet informatique quel que soit le langage à mon avis car cela permet de gérer des configurations complexes avec des variables/références tout en restant simple de format.
- la sauvegarde générique complète d'un projet ML du code au modèle et dataset me semble un point important dans l'optique de backup et versioning de projets ML en associant l'intégralité des ressources: code, configuration, modèle, tokenizer, dataset etc...
- le typage fort dans Python est devenu un outil intéressant qui permet d'améliorer la robustesse globale du code, de réduire la quantité de tests unitaires. Mypy semble être une solution robuste pour vérifier les types même s'il faut filtrer de nombreuses dépendances externes qui n'intègrent pas la gestion des types. Cependant, l'utilisation trop fréquente des unions de types dans les librairies Python peut conduire à des signatures de type assez indigestes.
- L'utilisation des types génériques et abstraits est fonctionnelle mais reste assez fastidieuse en Python et ne donne pas l'impression d'être une fonctionnalité native du langage (sans parler des cast au runtime qui peuvent poser des problèmes de performance). Il vaut mieux rester dans les patterns orienté-objet classiques et éviter de trop s'aventurer en dehors des sentiers battus.
- L'utilisation des NewTypes reste encore anecdotique de mon point de vue (en particulier, les opérations mathématiques ou de concaténation sur ces types leur font perdre leur spécificité)

Si vous avez des questions, n'hésitez pas à me contacter.
