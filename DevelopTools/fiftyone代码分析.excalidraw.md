---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠==


# Text Elements
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.VOCDetectionDataset,
    name=name,
) ^R56JVYUa

类方法的工厂函数，先构建VOCDetectionDataset
然后调用add_dir,然后返回构造好的数据集实例
 ^r6AqmKx3

又是一个工厂函数，通过 dataset 的 get_dataset_importer_cls 方法得到
对应的 importer 名称，然后构建 importer 实例。然后调用 import_samples
方法。
import_samples 方法使用构建好的 importer 实例 来导入 sample ^DTYC97po

importer 实际上是一个可迭代的上下文管理器 ^RYKXUTqY

import_samples 方法逻辑大略为：
1. 进入到 importer 的上下文
2. 遍历 importer，调用 dataset 的 add_samples 方法
3. 通过 importer 输出 dataset 的整体信息 ^pzVNW7DS

has_dataset_info
get_dataset_info ^po9WG0oC

%%
# Drawing
```json
{
	"type": "excalidraw",
	"version": 2,
	"source": "https://excalidraw.com",
	"elements": [
		{
			"type": "text",
			"version": 98,
			"versionNonce": 2098947544,
			"isDeleted": false,
			"id": "R56JVYUa",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -333.5272494229404,
			"y": -271.0994186401367,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 432,
			"height": 96,
			"seed": 531490666,
			"groupIds": [],
			"roundness": null,
			"boundElements": [],
			"updated": 1677548517285,
			"link": null,
			"locked": false,
			"fontSize": 16,
			"fontFamily": 3,
			"text": "dataset = fo.Dataset.from_dir(\n    dataset_dir=dataset_dir,\n    dataset_type=fo.types.VOCDetectionDataset,\n    name=name,\n)",
			"rawText": "dataset = fo.Dataset.from_dir(\n    dataset_dir=dataset_dir,\n    dataset_type=fo.types.VOCDetectionDataset,\n    name=name,\n)",
			"baseline": 92,
			"textAlign": "left",
			"verticalAlign": "top",
			"containerId": null,
			"originalText": "dataset = fo.Dataset.from_dir(\n    dataset_dir=dataset_dir,\n    dataset_type=fo.types.VOCDetectionDataset,\n    name=name,\n)"
		},
		{
			"type": "ellipse",
			"version": 30,
			"versionNonce": 438064042,
			"isDeleted": false,
			"id": "HYZGGxMG9sUttAoYjU4Xs",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -146.39996337890625,
			"y": -285.0812454223633,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 103.199951171875,
			"height": 48,
			"seed": 207215286,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [
				{
					"id": "iG5hOGBui_yt4WPlwuE08",
					"type": "arrow"
				}
			],
			"updated": 1677050022060,
			"link": null,
			"locked": false
		},
		{
			"type": "text",
			"version": 409,
			"versionNonce": 685882602,
			"isDeleted": false,
			"id": "r6AqmKx3",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -71.20001220703125,
			"y": -117.3812484741211,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 371,
			"height": 61,
			"seed": 1401880758,
			"groupIds": [],
			"roundness": null,
			"boundElements": [
				{
					"id": "iG5hOGBui_yt4WPlwuE08",
					"type": "arrow"
				}
			],
			"updated": 1677050048544,
			"link": null,
			"locked": false,
			"fontSize": 16,
			"fontFamily": 3,
			"text": "类方法的工厂函数，先构建VOCDetectionDataset\n然后调用add_dir,然后返回构造好的数据集实例\n",
			"rawText": "类方法的工厂函数，先构建VOCDetectionDataset\n然后调用add_dir,然后返回构造好的数据集实例\n",
			"baseline": 57,
			"textAlign": "left",
			"verticalAlign": "top",
			"containerId": null,
			"originalText": "类方法的工厂函数，先构建VOCDetectionDataset\n然后调用add_dir,然后返回构造好的数据集实例\n"
		},
		{
			"type": "arrow",
			"version": 855,
			"versionNonce": 57142698,
			"isDeleted": false,
			"id": "iG5hOGBui_yt4WPlwuE08",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -42.496993530208634,
			"y": -258.4283940029351,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 13.576437950276208,
			"height": 134.94713942529836,
			"seed": 230369706,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [],
			"updated": 1677050035898,
			"link": null,
			"locked": false,
			"startBinding": {
				"elementId": "HYZGGxMG9sUttAoYjU4Xs",
				"focus": -1.0076522975831452,
				"gap": 1
			},
			"endBinding": {
				"elementId": "r6AqmKx3",
				"focus": -0.739987852677838,
				"gap": 6.100006103515625
			},
			"lastCommittedPoint": null,
			"startArrowhead": null,
			"endArrowhead": "arrow",
			"points": [
				[
					0,
					0
				],
				[
					13.576437950276208,
					134.94713942529836
				]
			]
		},
		{
			"type": "ellipse",
			"version": 55,
			"versionNonce": 826408374,
			"isDeleted": false,
			"id": "yBFAQbBZou49hhK33tuoK",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -12,
			"y": -93.08123016357422,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 84.79998779296875,
			"height": 24.79998779296875,
			"seed": 271486890,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [
				{
					"id": "z2JjPK21GamJZNy3sMyHe",
					"type": "arrow"
				}
			],
			"updated": 1677050621894,
			"link": null,
			"locked": false
		},
		{
			"type": "text",
			"version": 421,
			"versionNonce": 1862031658,
			"isDeleted": false,
			"id": "DTYC97po",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -271.20001220703125,
			"y": 31.418739318847656,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 601,
			"height": 83,
			"seed": 662441130,
			"groupIds": [],
			"roundness": null,
			"boundElements": [
				{
					"id": "z2JjPK21GamJZNy3sMyHe",
					"type": "arrow"
				}
			],
			"updated": 1677050707398,
			"link": null,
			"locked": false,
			"fontSize": 16,
			"fontFamily": 3,
			"text": "又是一个工厂函数，通过 dataset 的 get_dataset_importer_cls 方法得到\n对应的 importer 名称，然后构建 importer 实例。然后调用 import_samples\n方法。\nimport_samples 方法使用构建好的 importer 实例 来导入 sample",
			"rawText": "又是一个工厂函数，通过 dataset 的 get_dataset_importer_cls 方法得到\n对应的 importer 名称，然后构建 importer 实例。然后调用 import_samples\n方法。\nimport_samples 方法使用构建好的 importer 实例 来导入 sample",
			"baseline": 79,
			"textAlign": "left",
			"verticalAlign": "top",
			"containerId": null,
			"originalText": "又是一个工厂函数，通过 dataset 的 get_dataset_importer_cls 方法得到\n对应的 importer 名称，然后构建 importer 实例。然后调用 import_samples\n方法。\nimport_samples 方法使用构建好的 importer 实例 来导入 sample"
		},
		{
			"type": "arrow",
			"version": 59,
			"versionNonce": 1491844010,
			"isDeleted": false,
			"id": "z2JjPK21GamJZNy3sMyHe",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": 45.93291877972888,
			"y": -64.44401729165338,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 181.93291877972888,
			"height": 88.96276271401666,
			"seed": 1483262646,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [],
			"updated": 1677050654317,
			"link": null,
			"locked": false,
			"startBinding": {
				"elementId": "yBFAQbBZou49hhK33tuoK",
				"focus": -0.9865228664969711,
				"gap": 4.795207434343492
			},
			"endBinding": {
				"elementId": "DTYC97po",
				"focus": -0.685784056352276,
				"gap": 6.899993896484375
			},
			"lastCommittedPoint": null,
			"startArrowhead": null,
			"endArrowhead": "arrow",
			"points": [
				[
					0,
					0
				],
				[
					-181.93291877972888,
					88.96276271401666
				]
			]
		},
		{
			"type": "ellipse",
			"version": 63,
			"versionNonce": 2111603178,
			"isDeleted": false,
			"id": "NHt_JmtKemafeYq3hJ1_f",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -12,
			"y": 47.718788146972656,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 138.4000244140625,
			"height": 30.39999389648437,
			"seed": 1207936438,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [
				{
					"id": "ttLFIErYWzh1kGZGQvFkO",
					"type": "arrow"
				}
			],
			"updated": 1677050811688,
			"link": null,
			"locked": false
		},
		{
			"type": "text",
			"version": 397,
			"versionNonce": 1751035647,
			"isDeleted": false,
			"id": "RYKXUTqY",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": 86.40008544921875,
			"y": -27.78124237060547,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 343,
			"height": 21,
			"seed": 2077632246,
			"groupIds": [],
			"roundness": null,
			"boundElements": [
				{
					"id": "ttLFIErYWzh1kGZGQvFkO",
					"type": "arrow"
				}
			],
			"updated": 1677051124929,
			"link": null,
			"locked": false,
			"fontSize": 16,
			"fontFamily": 3,
			"text": "importer 实际上是一个可迭代的上下文管理器",
			"rawText": "importer 实际上是一个可迭代的上下文管理器",
			"baseline": 17,
			"textAlign": "left",
			"verticalAlign": "top",
			"containerId": null,
			"originalText": "importer 实际上是一个可迭代的上下文管理器"
		},
		{
			"type": "arrow",
			"version": 232,
			"versionNonce": 1540257567,
			"isDeleted": false,
			"id": "ttLFIErYWzh1kGZGQvFkO",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": 107.08464147816058,
			"y": 46.81559147468964,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 75.74601396913123,
			"height": 48.29684605232636,
			"seed": 705050166,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [],
			"updated": 1677051124929,
			"link": null,
			"locked": false,
			"startBinding": {
				"elementId": "NHt_JmtKemafeYq3hJ1_f",
				"focus": 0.3448612823365828,
				"gap": 6.261427749522811
			},
			"endBinding": {
				"elementId": "RYKXUTqY",
				"focus": 0.26754415034386675,
				"gap": 5.29998779296875
			},
			"lastCommittedPoint": null,
			"startArrowhead": null,
			"endArrowhead": "arrow",
			"points": [
				[
					0,
					0
				],
				[
					75.74601396913123,
					-48.29684605232636
				]
			]
		},
		{
			"type": "text",
			"version": 264,
			"versionNonce": 1016310570,
			"isDeleted": false,
			"id": "pzVNW7DS",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -506.79998779296875,
			"y": 204.01874542236328,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 447,
			"height": 83,
			"seed": 795808298,
			"groupIds": [],
			"roundness": null,
			"boundElements": [
				{
					"id": "w1PzpnkNi-V1dDJYiw1tv",
					"type": "arrow"
				}
			],
			"updated": 1677051067773,
			"link": null,
			"locked": false,
			"fontSize": 16,
			"fontFamily": 3,
			"text": "import_samples 方法逻辑大略为：\n1. 进入到 importer 的上下文\n2. 遍历 importer，调用 dataset 的 add_samples 方法\n3. 通过 importer 输出 dataset 的整体信息",
			"rawText": "import_samples 方法逻辑大略为：\n1. 进入到 importer 的上下文\n2. 遍历 importer，调用 dataset 的 add_samples 方法\n3. 通过 importer 输出 dataset 的整体信息",
			"baseline": 79,
			"textAlign": "left",
			"verticalAlign": "top",
			"containerId": null,
			"originalText": "import_samples 方法逻辑大略为：\n1. 进入到 importer 的上下文\n2. 遍历 importer，调用 dataset 的 add_samples 方法\n3. 通过 importer 输出 dataset 的整体信息"
		},
		{
			"type": "rectangle",
			"version": 145,
			"versionNonce": 345328182,
			"isDeleted": false,
			"id": "5cViIjaWPwIGh6L9PH0j5",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -481.8065622472153,
			"y": 264.39623408437626,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 337.28726628153237,
			"height": 24.92562003995505,
			"seed": 1008677494,
			"groupIds": [],
			"roundness": {
				"type": 3
			},
			"boundElements": [
				{
					"id": "YpI9YS3je3Zn6ec-c_KxD",
					"type": "arrow"
				}
			],
			"updated": 1677051015492,
			"link": null,
			"locked": false
		},
		{
			"type": "arrow",
			"version": 145,
			"versionNonce": 1200750070,
			"isDeleted": false,
			"id": "YpI9YS3je3Zn6ec-c_KxD",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -448.46892244374965,
			"y": 294.5179165680005,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 25.155026305609397,
			"height": 39.66188557578613,
			"seed": 737445482,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [],
			"updated": 1677051042483,
			"link": null,
			"locked": false,
			"startBinding": {
				"elementId": "5cViIjaWPwIGh6L9PH0j5",
				"focus": 0.7011683872292686,
				"gap": 5.196062443669206
			},
			"endBinding": {
				"elementId": "po9WG0oC",
				"focus": 0.3036952927396523,
				"gap": 5.927213651158013
			},
			"lastCommittedPoint": null,
			"startArrowhead": null,
			"endArrowhead": "arrow",
			"points": [
				[
					0,
					0
				],
				[
					-25.155026305609397,
					39.66188557578613
				]
			]
		},
		{
			"type": "text",
			"version": 57,
			"versionNonce": 458717302,
			"isDeleted": false,
			"id": "po9WG0oC",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -592.1742262598541,
			"y": 340.10701579494463,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 152,
			"height": 38,
			"seed": 1344472810,
			"groupIds": [],
			"roundness": null,
			"boundElements": [
				{
					"id": "YpI9YS3je3Zn6ec-c_KxD",
					"type": "arrow"
				}
			],
			"updated": 1677051050203,
			"link": null,
			"locked": false,
			"fontSize": 16,
			"fontFamily": 3,
			"text": "has_dataset_info\nget_dataset_info",
			"rawText": "has_dataset_info\nget_dataset_info",
			"baseline": 35,
			"textAlign": "left",
			"verticalAlign": "top",
			"containerId": null,
			"originalText": "has_dataset_info\nget_dataset_info"
		},
		{
			"type": "ellipse",
			"version": 30,
			"versionNonce": 111429738,
			"isDeleted": false,
			"id": "S5dh6dfnUoNc-BL73fDpR",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -279.62996913987865,
			"y": 90.42426276630493,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 154.51628107173042,
			"height": 33.051602833619825,
			"seed": 905771446,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [
				{
					"id": "w1PzpnkNi-V1dDJYiw1tv",
					"type": "arrow"
				}
			],
			"updated": 1677051067773,
			"link": null,
			"locked": false
		},
		{
			"type": "arrow",
			"version": 62,
			"versionNonce": 1525828406,
			"isDeleted": false,
			"id": "w1PzpnkNi-V1dDJYiw1tv",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -212.7004860099844,
			"y": 125.95472950835335,
			"strokeColor": "#c92a2a",
			"backgroundColor": "transparent",
			"width": 115.68060991766947,
			"height": 66.10320566723965,
			"seed": 1142738038,
			"groupIds": [],
			"roundness": {
				"type": 2
			},
			"boundElements": [],
			"updated": 1677051067773,
			"link": null,
			"locked": false,
			"startBinding": {
				"elementId": "S5dh6dfnUoNc-BL73fDpR",
				"focus": -0.2779549499316681,
				"gap": 2.626128183127456
			},
			"endBinding": {
				"elementId": "pzVNW7DS",
				"focus": -0.468172312592767,
				"gap": 11.96081024677028
			},
			"lastCommittedPoint": null,
			"startArrowhead": null,
			"endArrowhead": "arrow",
			"points": [
				[
					0,
					0
				],
				[
					-115.68060991766947,
					66.10320566723965
				]
			]
		}
	],
	"appState": {
		"theme": "light",
		"viewBackgroundColor": "#ffffff",
		"currentItemStrokeColor": "#000000",
		"currentItemBackgroundColor": "transparent",
		"currentItemFillStyle": "hachure",
		"currentItemStrokeWidth": 1,
		"currentItemStrokeStyle": "solid",
		"currentItemRoughness": 1,
		"currentItemOpacity": 100,
		"currentItemFontFamily": 3,
		"currentItemFontSize": 16,
		"currentItemTextAlign": "left",
		"currentItemStartArrowhead": null,
		"currentItemEndArrowhead": "arrow",
		"scrollX": 622.2961613144086,
		"scrollY": 654.1405239046184,
		"zoom": {
			"value": 1.1
		},
		"currentItemRoundness": "round",
		"gridSize": null,
		"colorPalette": {},
		"currentStrokeOptions": null,
		"previousGridSize": null
	},
	"files": {}
}
```
%%