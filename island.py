import random
class prompt():
    def __init__(self, text, acc):
        self.text = text
        self.acc = acc


class island():
    def __init__(self):
        self.texts = []
        self.accs = []
    def add_prompt(self, prompt):
        self.texts.append(prompt.text)
        self.accs.append(prompt.acc)
    def sort_prompts(self):
        combined = list(zip(self.texts, self.accs))
        combined.sort(key=lambda x: x[1], reverse=True)
        text_list, acc_list = zip(*combined)
        return text_list, acc_list
    def get_prompts(self):
        rand_idx = random.sample(range(len(self.texts)), 2)
        # if self.accs[rand_idx[0]] > self.accs[rand_idx[1]]:
        #     rand_idx[0], rand_idx[1] = rand_idx[1], rand_idx[0]
        return self.texts[rand_idx[0]] + ' \n' + self.texts[rand_idx[1]] + ' \n'
    def get_prompts_fun(self):
        combined = list(zip(self.texts, self.accs))
        # 按照 accs 排序
        combined.sort(key=lambda x: x[1], reverse=True)
        # 返回 accs 最大的两个值对应的 texts
        return combined[1][0] + '\n' + combined[0][0] + '\n'
    def transfer(self):
        text_list, acc_list = self.sort_prompts()
        self.texts = [text_list[0], text_list[1]]
        self.accs = [acc_list[0], acc_list[1]]
    def is_repeat(self, text):
        return text in self.texts
    def update(self,new_islands):
        self.texts = new_islands.texts
        self.accs = new_islands.accs
    def remove(self):
        if len(self.texts) > 6:
            combined = list(zip(self.accs, self.texts))
            sorted_combined = sorted(combined, key=lambda x: x[0])
            top_6 = sorted_combined[-6:]
            self.accs, self.texts = map(list, zip(*top_6))
    def remove_lmx(self):
        while len(self.texts) > 6:
            rand_idx = random.sample(range(len(self.texts)), 2)
            if self.accs[rand_idx[0]] > self.accs[rand_idx[1]]:
                del_idx = rand_idx[1]
            else:
                del_idx = rand_idx[0]
            del self.texts[del_idx]
            del self.accs[del_idx]

class island_group():
    def __init__(self):
        self.islands = []
    def add_island(self, island):
        self.islands.append(island)
    def sort_transfer_islands(self):
        # 根据 island.accs 的最大值对 islands 列表进行排序
        self.islands.sort(key=lambda island: max(island.accs), reverse=True)

        # 计算 "good" 和 "bad" 的分界点
        split_point = len(self.islands) // 2

        # 将 islands 列表分为 "good" 和 "bad" 两部分
        good_islands = self.islands[:split_point]
        bad_islands = self.islands[split_point:]

        # 返回 "good" 和 "bad" 的 island 的索引
        good_indices = [self.islands.index(island) for island in good_islands]
        bad_indices = [self.islands.index(island) for island in bad_islands]
        for bad_index in bad_indices:
            # 清空 bad_island 的 text 和 accs
            self.islands[bad_index].text = []
            self.islands[bad_index].accs = []

            # 从 good_indices 中随机选择一个 island
            good_index = random.choice(good_indices)

            # 复制 good_island 的前两个个体
            tex, acc = self.islands[good_index].transfer()
            self.islands[bad_index].text.extend(tex)
            self.islands[bad_index].accs.extend(acc)
    def island_update(self):
        acc_list = []
        for island in self.islands:
            acc_list.append(max(island.accs))
        median = sorted(acc_list)[len(acc_list) // 2]
        lower_half_indices = [i for i, acc in enumerate(acc_list) if acc < median]
        upper_half_indices = [i for i, acc in enumerate(acc_list) if acc >= median]
        for i, j in zip(lower_half_indices, upper_half_indices):
            self.islands[i].update(self.islands[j])
     