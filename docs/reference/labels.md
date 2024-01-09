# Label Targeters
A label is the target/ground truth associated with a single sample. This class is a utility containing algorithms for generating or updating target labels in a variety of formats to include:

- Single, fixed integer
- Single, fixed string
- Random label from a predefined list
- Fixed integer offset
- Fixed values as specified by an ordered list
- Exact value as the input label
- Replacement of object detection labels with fixed integer
- Transcript from a fixed list with the length closest to that of the input label

::: armory.labels