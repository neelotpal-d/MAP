# MAP backend release attribution

Backend algorithms and integration: https://github.com/neelotpal-d/MAP
These are the exact five tested backend component folders, including recorded
maintenance patches. No GUI source, training checkpoints or user jobs are included.
The companion interface and existing FEA retain their MultiAxisPrinting authorship:
https://github.com/RyanTaoLiu/MultiAxisPrinting/tree/dev2

The files in this release asset are authoritative; the release tag is a repository
coordination point and its automatic source archives may differ from this snapshot.
No new blanket licence or reassignment of third-party ownership is made.

## Retained project notices

# Third-party notices

- `*/runtime/siren_pytorch/`: embedded SIREN-PyTorch implementation, Phil Wang
  (lucidrains), copyright 2020, MIT. Licence copies are retained beside the code
  and in `third_party/siren-pytorch/LICENSE`.
  Source: https://github.com/lucidrains/siren-pytorch
- `CurvedSupportGenerator/runtime/SupportGenerationCurved/bambu_sizing.py`:
  selected calculations adapted from Bambu Lab BambuStudio commit
  `79df7ae309be398ed30c9f397c13c9911c2edc57`, AGPL-3.0. Original notice is retained;
  licence and source links are beside that module and in `third_party/BambuStudio/`.
- `standalone_bambu_layer_slicer/`: Bambu printer reference/template inputs
  retain their vendor metadata and attribution. The project authorship notice
  refers to assembly/export code, not ownership of Bambu firmware, metadata,
  presets or logos. The A1 reference archive and H2D templates are required
  packaging inputs, not generated user print jobs.
- `MultiAxisPrinting-dev/`: existing upstream interface code/resources; existing
  notices retained. Upstream repository/author/licence details must be completed
  by the maintainer. No new sole-author claim is attached to this directory.
- Dependencies are installed separately and retain their own licences. They
  include Qt/PyQt, PyQtAds, NumPy/SciPy, PyVista/VTK, PyMeshLab, libigl,
  Shapely/GEOS, TetGen, Torch, and the packages in `environment.yml`.

Repository attribution does not grant a blanket licence or transfer
third-party rights. See `LICENSE_STATUS.md`.


# Licence information to complete before public release

The source snapshot did not contain a project-wide licence or a verified licence
for the original interface. This package preserves existing notices and adds
the identified SIREN MIT and Bambu AGPL licence texts; it does not choose a new
licence for the author's other work or override existing terms.

- Project licence: <!-- maintainer: select applicable licence / add LICENSE -->
- Original interface provenance and licence: <!-- maintainer: add verified details -->
- Resource/logo and printer-reference redistribution details: <!-- add details -->

The Bambu-derived sizing module already declares AGPL-3.0. Its attribution and
terms remain applicable. This source layout is prepared for version control;
public-release licensing metadata remains incomplete until the maintainer fills
these items. No GitHub repository has been created or pushed.
