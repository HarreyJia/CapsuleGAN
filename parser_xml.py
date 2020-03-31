from xml.etree import ElementTree as ET

tree = ET.parse("../../Downloads/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-30178/3000566-03192/069.xml")

root = tree.getroot()

NAMESPACE = "{http://www.nih.gov}"

for node in root.iter(NAMESPACE + 'readingSession'):
    # 节点的标签名称和内容
    print(node.tag, node.text)
    for i in node:
        if i.tag == NAMESPACE + "unblindedReadNodule":
            print(i)