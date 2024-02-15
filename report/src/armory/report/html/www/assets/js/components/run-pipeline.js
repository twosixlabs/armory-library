import { computed, watch } from 'vue';

const DownArrow = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M16 19 12 23m0 0-4-4M12 23V0" />
        </svg>
    `,
};

const DownLine = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
            <path stroke-linecap="butt" stroke-linejoin="butt" d="M12 24V0" />
        </svg>
    `,
};

const TopLeftCorner = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 112 24" stroke-width="2" stroke="currentColor" class="w-28 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M56 24V1H112" />
        </svg>
    `,
};

const TopGap = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 32 2" stroke-width="2" stroke="currentColor" class="w-8 h-0.5">
            <path stroke-linecap="round" stroke-linejoin="round" d="M0 1H32" />
        </svg>
    `,
};

const TopMiddle = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 112 24" stroke-width="2" stroke="currentColor" class="w-28 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M0 1H112M56 1V24" />
        </svg>
    `,
};

const TopRightCorner = {
    template: `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 112 24" stroke-width="2" stroke="currentColor" class="w-28 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" d="M56 24V1H0" />
        </svg>
    `,
};

const Split = {
    props: {
        num: Number,
    },
    components: {
        TopGap,
        TopMiddle,
        TopLeftCorner,
        TopRightCorner,
    },
    template: `
        <div class="flex">
            <TopLeftCorner></TopLeftCorner>
            <template v-for="i in num-2">
                <TopGap></TopGap>
                <TopMiddle></TopMiddle>
            </template>
            <TopGap v-if="num > 2"></TopGap>
            <TopRightCorner></TopRightCorner>
        </div>
        <div class="flex gap-8">
            <slot></slot>
        </div>
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
        Split,
    },
    setup(props) {
        const numChains = computed(() => Object.keys(props.run.evaluation.perturbations).length);
        return { numChains };
    },
    template: `
        <div v-if="run.evaluation" class="flex flex-row justify-center mt-2">
            <div class="flex flex-col items-center">
                <Box :name="run.evaluation.dataset.name" output type="dataset">
                </Box>
                <Split :num="numChains">
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
                        <Box :name="run.evaluation.model.name" input type="model"></Box>
                    </Chain>
                </Split>
            </div>
            <div class="border-l-2 mx-4"></div>
            <div class="flex flex-col">
                <div class="flex gap-2 items-center justify-center">
                    <select class="select select-bordered select-sm w-40">
                        <option disabled>First chain?</option>
                        <option selected>benign</option>
                        <option>attacked</option>
                        <option>defended</option>
                    </select>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_0_benign_no_defense.png" />
                        Label: 35
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_4_benign_no_defense.png" />
                        Label: 9
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        Label: 38
                    </div>
                </div>
                <div class="border-t-2 my-2"></div>
                <div class="flex gap-2 items-center justify-center">
                    <select class="select select-bordered select-sm w-40">
                        <option disabled>First chain?</option>
                        <option>benign</option>
                        <option selected>attacked</option>
                        <option>defended</option>
                    </select>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_0_benign_no_defense.png" />
                        <span class="text-red-600">Label: 7</span>
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_4_benign_no_defense.png" />
                        <span class="text-red-600">Label: 16</span>
                    </div>
                    <div class="flex flex-col items-center">
                        <img alt="benign" src="./assets/img/batch_14_ex_5_benign_no_defense.png" />
                        <span class="text-green-600">Label: 38</span>
                    </div>
                </div>
            </div>
        </div>
        <div v-if="!run.evaluation" class="bg-red-100 border-2 border-red-200 flex justify-center my-10 mx-40 p-4 rounded-md">
            The selected run does not have an evaluation pipeline definition.
        </div>
    `,
};
