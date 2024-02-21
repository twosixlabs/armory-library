import { computed } from 'vue';
import { argmax } from '../utils/argmax.js';

export default {
    props: {
        batch: Number,
        lhsChain: String,
        rhsChain: String,
        run: Object,
        sample: Number,
        sideBySide: Boolean,
    },
    setup(props) {
        const lhsImage = computed(() => {
            const file = props.run.artifacts[props.lhsChain]?.[props.batch]?.[props.sample]?.file;
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });
        const lhsTarget = computed(() => 
            props.run.artifacts[props.lhsChain]?.[props.batch]?.[props.sample]?.targets
        );
        const lhsPrediction = computed(() => {
            const preds = props.run.artifacts[props.lhsChain]?.[props.batch]?.[props.sample]?.predictions;
            if (Array.isArray(preds)) {
                return argmax(preds);
            }
            return preds;
        });

        const rhsImage = computed(() => {
            const file = props.run.artifacts[props.rhsChain]?.[props.batch]?.[props.sample]?.file;
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });
        const rhsTarget = computed(() => 
            props.run.artifacts[props.rhsChain]?.[props.batch]?.[props.sample]?.targets
        );
        const rhsPrediction = computed(() => {
            const preds = props.run.artifacts[props.rhsChain]?.[props.batch]?.[props.sample]?.predictions;
            if (Array.isArray(preds)) {
                return argmax(preds);
            }
            return preds;
        });

        return {
            lhsImage,
            lhsPrediction,
            lhsTarget,
            rhsImage,
            rhsPrediction,
            rhsTarget,
        };
    },
    template: `
        <div class="flex gap-4 items-center justify-center">
            <div class="flex flex-col">
                <span>{{ lhsChain }}</span>
                <span>Target: {{ lhsTarget }}</span>
                <span>Prediction: {{ lhsPrediction }}</span>
            </div>
            <template v-if="lhsImage && rhsImage">
                <div v-if="!sideBySide" class="diff aspect-square max-w-xl">
                    <div class="diff-item-1">
                        <img :src="rhsImage" />
                    </div>
                    <div class="diff-item-2">
                        <img :src="lhsImage" />
                    </div>
                    <div class="diff-resizer"></div>
                </div>
                <div v-else class="flex gap-2">
                    <img :src="lhsImage" />
                    <img
                        :src="rhsImage"
                        class="border-l-2 pl-2"
                    />
                </div>
            </template>
            <div class="flex flex-col">
                <span>{{ rhsChain }}</span>
                <span>Target: {{ rhsTarget }}</span>
                <span>Prediction: {{ rhsPrediction }}</span>
            </div>
        </div>
    `,
};
