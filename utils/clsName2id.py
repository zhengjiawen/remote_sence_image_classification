from config import config


def readRelation(path):
    '''
    生成关系label映射，label为中文
    :param path:映射文件路径
    :return:dict, label 2 id and id 2 label
    '''
    claName2Id = {}
    claId2Name = {}

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            # line: 旱地:dry-field:1
            parts = line.strip().split(':')
            claName2Id[parts[0]] = parts[2]
            claId2Name[parts[2]] = parts[0]

    return claName2Id, claId2Name

# prepare relation dict
claName2Id, claId2Name = readRelation(config.relation_path)


def getIdByName(name):
    return claName2Id[name]

def getNameById(id):
    return claId2Name[id]




if __name__ == '__main__':
    path = 'D:/rssrai2019_scene_classification/ClsName2id.txt'

    name2Id, Id2Name = readRelation(path)

    print(name2Id)
