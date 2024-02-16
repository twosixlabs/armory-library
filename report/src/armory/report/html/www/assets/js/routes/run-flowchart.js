import { computed } from 'vue';

const Paths = {
    props: {
        height: Number,
        paths: Object,
        width: Number,
    },
    setup(props) {
        const classes = computed(() => [
            `h-[${props.height}px]`,
            `w-[${props.width}px]`,
        ]);
        const viewbox = computed(() => `0 0 ${props.width} ${props.height}`);
        return { classes, viewbox };
    },
    template: `
        <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            :viewBox="viewbox"
            stroke-width="2"
            stroke="currentColor"
            :class="classes">
            <path
                v-for="path in paths"
                stroke-linecap="round"
                stroke-linejoin="round"
                :d="paths"
            />
        </svg>
    `,
};

const DownArrow = {
    components: {
        Paths,
    },
    props: {
        arrowHeadLength: {
            default: 4,
            type: Number,
        },
        height: {
            default: 24,
            type: Number,
        },
        width: {
            default: 10,
            type: Number,
        }
    },
    setup(props) {
        const paths = computed(() => {
            const len = props.arrowHeadLength;
            const mid = props.width / 2;
            const arrowLeft = mid - len;
            const arrowRight = mid + len;
            const bottom = props.height - 1;
            const arrowTop = bottom - len;
            return [
                `M ${arrowLeft} ${arrowTop} ${mid} ${bottom}`,
                `M ${arrowRight} ${arrowTop} ${mid} ${bottom}`,
                `M ${mid} ${bottom} V 0`,
            ];
        });
        return { paths };
    },
    template: `
        <Paths :height="height" :paths="paths" :width="width" />
    `,
};

const DownLine = {
    components: {
        Paths,
    },
    props: {
        height: {
            default: 24,
            type: Number,
        },
        width: {
            default: 4,
            type: Number,
        }
    },
    setup(props) {
        const paths = computed(() => [`M ${props.width / 2} 0 V ${props.height}`]);
        return { paths };
    },
    template: `
        <Paths :height="height" :paths="paths" :width="width" />
    `,
};

const Split = {
    components: {
        Paths,
    },
    props: {
        boxWidth: Number,
        gapWidth: Number,
        height: Number,
        invert: {
            default: false,
            type: Boolean,
        },
        num: Number,
    },
    setup(props) {
        const width = computed(() =>
            props.boxWidth * props.num + props.gapWidth * (props.num - 1)
        );
        const paths = computed(() => {
            const halfBoxWidth = props.boxWidth / 2;
            const lineHeight = props.invert ? props.height - 1 : 1;

            const horizontal = `M ${halfBoxWidth} ${lineHeight} H ${width.value - halfBoxWidth}`;
            const verticals = [...Array(props.num)].map((_, index) => (
                halfBoxWidth + (props.boxWidth + props.gapWidth) * index
            )).map((x) => `M ${x} 1 V ${props.height - 1}`);

            return [horizontal, ...verticals];
        });
        return { paths, width };
    },
    template: `
        <Paths
            :height="height"
            :paths="paths"
            :width="width"
        />
    `,
};

const LineToBottom = {
    components: {
        Paths,
    },
    props: {
        boxHeight: {
            default: 112,
            type: Number,
        },
        gapHeight: {
            default: 24,
            type: Number,
        },
        fill: Number,
        width: {
            default: 4,
            type: Number,
        },
    },
    setup(props) {
        const height = computed(() =>
            (props.boxHeight + props.gapHeight) * props.fill
        );
        const paths = computed(() => [
            `M ${props.width / 2} 0 V ${height.value}`,
        ]);
        return { height, paths };
    },
    template: `
        <Paths
            :height="height"
            :paths="paths"
            :width="width"
        />
    `,
};

const Parallel = {
    components: {
        Split,
    },
    props: {
        boxWidth: {
            default: 112,
            type: Number,
        },
        gapWidth: {
            default: 32,
            type: Number,
        },
        join: {
            default: false,
            type: Boolean,
        },
        num: Number,
        splitHeight: {
            default: 24,
            type: Number,
        },
    },
    setup(props) {
        const gap = computed(() => `gap-[${props.gapWidth}px]`);
        return { gap };
    },
    template: `
        <Split
            :boxWidth="boxWidth"
            :gapWidth="gapWidth"
            :height="splitHeight"
            :num="num"
        />
        <div class="flex" :class="gap">
            <slot></slot>
        </div>
        <Split
            v-if="join"
            :boxWidth="boxWidth"
            :gapWidth="gapWidth"
            :height="splitHeight"
            :num="num"
            invert
        />
    `,
};


const Chain = {
    props: {
        name: String,
    },
    template: `
        <div class="flex flex-col items-center">
            <div class="tooltip" :data-tip="name">
                <div class="max-w-28 overflow-hidden text-ellipsis">
                    {{ name }}
                </div>
            </div>
            <slot></slot>
        </div>
    `,
};

const Box = {
    props: {
        input: {
            type: Boolean,
            default: false,
        },
        name: String,
        type: String,
        output: {
            type: Boolean,
            default: false,
        },
    },
    components: {
        DownArrow,
        DownLine,
    },
    setup(props) {
        const classes = computed(() => ({
            'bg-violet-300': props.type == "dataset",
            'border-violet-400': props.type == "dataset",
            'bg-red-300': props.type == "perturbation",
            'border-red-400': props.type == "perturbation",
            'bg-blue-300': props.type == "model",
            'border-blue-400': props.type == "model",
            'bg-yellow-200': props.type == "metric",
            'border-yellow-300': props.type == "metric",
        }));
        return { classes };
    },
    template: `
        <div class="flex flex-col items-center w-28">
            <DownArrow v-if="input"></DownArrow>
            <div class="tooltip tooltip-right" :data-tip="name">
                <div :class="classes" class="border-2 flex flex-col items-center justify-center h-28 p-2 rounded-lg shadow w-28">
                    <span class="text-xs uppercase">
                        {{ type }}
                    </span>
                    <span class="max-h-24 overflow-hidden text-ellipsis w-24">
                        {{ name }}
                    </span>
                </div>
            </div>
            <DownLine v-if="output"></DownLine>
        </div>
    `,
};

export default {
    props: {
        run: Object,
    },
    components: {
        Box,
        Chain,
        LineToBottom,
        Parallel,
    },
    setup(props) {
        const numChains = computed(() => Object.keys(props.run.evaluation.perturbations).length);
        const maxNumPerturbations = computed(
            () => Math.max(...Object.values(props.run.evaluation.perturbations).map(p => p.length))
        );
        return { maxNumPerturbations, numChains };
    },
    template: `
        <div v-if="run.evaluation" class="flex flex-row justify-center my-2">
            <div class="flex flex-col items-center">
                <Box :name="run.evaluation.dataset.name" output type="dataset">
                </Box>
                <Parallel join :num="numChains">
                    <Chain
                        v-for="(perturbations, chain) of run.evaluation.perturbations"
                        :key="chain"
                        :name="chain"
                    >
                        <Box
                            v-for="{ name } in perturbations"
                            :key="name"
                            :name="name"
                            input
                            type="perturbation"
                        ></Box>
                        <LineToBottom :fill="maxNumPerturbations - perturbations.length">
                        </LineToBottom>
                    </Chain>
                </Parallel>
                <Box :name="run.evaluation.model.name" input output type="model"></Box>
                <Parallel :num="3" :splitHeight="2">
                    <Box input type="metric" name="accuracy" />
                    <Box input type="metric" name="accuracy" />
                    <Box input type="metric" name="accuracy" />
                </Parallel>
            </div>
        </div>
        <div v-if="!run.evaluation" class="bg-red-100 border-2 border-red-200 flex justify-center my-10 mx-40 p-4 rounded-md">
            The selected run does not have an evaluation pipeline definition.
        </div>
    `,
};
